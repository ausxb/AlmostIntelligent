module FFNN (
        Layer(..), Spec(..),
        sigma, dfSigma, tanh, dfTanh,
        xEntropyLoss, dfXEntropyLoss,
        quadLoss, dfQuadLoss,
        xavier, uniInvRoot, makeLayers,
        iteration, checkLayers,
        inferBatch, trainNetwork,
        testBinaryClassifier, testUsingComparator,
        compareBinary, compareByMax, shuffleSplit,
        printLayer, printBatch
    ) where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Control.Monad
import Control.Monad.Primitive
import Control.Monad.State.Strict

import Data.List.Split (chunksOf)
import qualified Data.Vector as Vec
        (Vector, empty, toList, splitAt,
         length, freeze, thaw, foldr')
import qualified Data.Vector.Mutable as MVec
        (MVector, swap)

import System.Random.MWC
import Statistics.Distribution
import Statistics.Distribution.Normal
import Statistics.Distribution.Uniform

import Text.Printf

import Prelude (
        IO, Floating, Int, Bool,
        map, foldl, foldl1, mapM,
        zip, unzip, repeat, head, 
        snd, uncurry,
        fromIntegral, print, putChar,
        sum, sqrt, log, exp,
        (+), (-), (/), (*),
        ($), (.), (==), (>), (<), (&&)
    )

-- A structure containing needed components for the 
-- forward and backward passes below.
-- dfActv is the "gradient" of vectorized activation function.
-- It is a column vector instead of a true Jacobian and is
-- intended for use with the Hadamard product.
data Layer = Layer {
    w      :: Matrix R,
    b      :: Vector R,
    actv   :: Matrix R -> Matrix R,
    dfActv :: Matrix R -> Matrix R
}

-- Provides a specification for each layer to be constructed.
-- m is the output dimension, n is the input dimension
-- fn is the activation function, dfn is the derivative of fn
-- winit is the weight initialization method. winit takes in the
-- layer dimensions and must produce an instance of ContGen,
-- a continuous distribution. Note that fn and dfn are scalar
-- valued functions that are then "vectorized" with cmap
-- during layer construction.
data Spec d = Spec {
    m     :: Int,
    n     :: Int,
    fn    :: R -> R,
    dfn   :: R -> R,
    winit :: Int -> Int -> d
}

type FwdAcc  = [(Matrix R, Matrix R, Layer)]
type BackAcc = [Layer]

type Extractor a = (a -> Vector R)
type Comparator  = (Vector R
                    -> Vector R -> Bool)
type Counter   m = (Comparator -> Vector R
                    -> Vector R -> m -> m)

class Metrics m where
    count :: m -> Int
    
data SimpleMetrics = SimpleMetrics {
    simpleMetricsCount :: Int,
    correct            :: Int,
    wrong              :: Int
}

instance Metrics SimpleMetrics where
    count = simpleMetricsCount

data BinaryMetrics = BinaryMetrics {
    binaryMetricsCount :: Int,
    correct_0          :: Int,
    wrong_0            :: Int,
    correct_1          :: Int,
    wrong_1            :: Int
}

instance Metrics BinaryMetrics where
    count = binaryMetricsCount

{---------------- Network Configuration ----------------}

sigma :: Floating t => t -> t
sigma z = e/(e + 1)
    where e = exp z
          
dfSigma :: Floating t => t -> t
dfSigma z = c * (1 - c)
    where c = sigma z

tanh :: Floating t => t -> t
tanh z = (e - 1)/(e + 1)
    where e = exp (2 * z)

dfTanh :: Floating t => t -> t
dfTanh z = 1 - c * c
    where c = tanh z
          
xEntropyLoss :: Matrix R -> Matrix R -> R
xEntropyLoss y a = (sumElements ((y * ln_a)
                        + (not_y * ln_not_a)))
                        / (fromIntegral (-n))
    where n = cols y
          not_y = cmap (1 -) y
          ln_a = cmap log a
          ln_not_a = cmap (log . (1 -)) a

dfXEntropyLoss :: Matrix R -> Matrix R -> Matrix R
dfXEntropyLoss y s = cmap (/ fromIntegral (cols y))
                          ((s - y) / (s * (cmap (1 -) s)))
{-    repmat cxe 1 (cols y)
    where cxe = asColumn
                $ cmap (/ fromIntegral (cols y))
                $ sumColumns
                $ (s - y) / (s * (cmap (1 -) s)) -}

quadLoss :: Matrix R -> Matrix R -> R
quadLoss y a = (sum (map normsq diffs)) / (2 * n)
    where diffs = toColumns (a - y)
          normsq x = dot x x
          n = fromIntegral $ cols y

dfQuadLoss :: Matrix R -> Matrix R -> Matrix R
dfQuadLoss y a = cmap (/ fromIntegral (cols y))
                      (a - y)

sumColumns :: Matrix R -> Vector R
sumColumns = foldl1 (+) . toColumns

sumRows :: Matrix R -> Vector R
sumRows = foldl1 (+) . toRows



{---------------- Network Training ----------------}

-- Take outer products of corresponding columns and adds the results.
-- Let U = [u_1 u_2 ... u_n] and V = [v_1 v_2 ... v_n] where
-- the u_i and v_i are columns of U and V respectively. Then
-- stackOuter gives (u_1 x v_1) + (u_2 x v_2) + ... + (u_n x v_n)
-- where x denotes the outer product.
stackOuter :: Matrix R -> Matrix R -> Matrix R
stackOuter dz a = foldl1 (+) $ map (uncurry outer) (zip dzcols acols)
    where dzcols = toColumns dz
          acols = toColumns a

-- Takes a layer, produces bindable State processor. Accumulates
-- a (weighted input, previous activation, layer) triple.
advanceLayer :: Layer -> (FwdAcc -> State (Matrix R) FwdAcc)
advanceLayer l = \acc -> state ( \x ->
                    let z = (_W <> x)
                            + (repmat (asColumn _b) 1 (cols x))
                    in ((z, x, l) : acc, actv l $ z) )
    where _W = w l
          _b = b l

-- Returned FwdAcc has layers in reverse order because of list construction,
-- which makes definition and application of backwardPass straightforward.
forwardPass :: [Layer] -> State (Matrix R) FwdAcc
forwardPass = foldl (>>=) (state $ \x -> ([], x)) . map advanceLayer

-- Takes a learning rate and performs one step of backpropagation.
-- The state processor takes the backpropagated errors from the
-- previous layer. It then computes the gradients for the current
-- layer using the stored matrix of weighted inputs from the batch.
-- The gradients are then multiplied by the tranpose of the weight
-- matrix and backpropagated to the next state processor.
updateLayer :: R -> (Matrix R, Matrix R, Layer)
                 -> (BackAcc -> State (Matrix R) BackAcc)
updateLayer eta (z, x, l) = \acc -> state (
        \bp -> let dz = bp * (dfActv l $ z)
                   _W = (-) (w l) $ cmap (* eta) $ stackOuter dz x
                   _b = (-) (b l) $ cmap (* eta) $ sumColumns dz
                   bp' = tr (w l) <> dz
               in (l{ w = _W, b = _b } : acc, bp') )

-- The reversed list from the forward pass is reversed again, yielding
-- an updated list of layers in the original order.
backwardPass :: R -> FwdAcc -> State (Matrix R) BackAcc
backwardPass eta = foldl (>>=) (state $ \x -> ([], x)) . map (updateLayer eta)

-- Performs forward and backward pass over a batch. A learning rate and
-- gradient of a cost function is required. For averaging cost, the
-- scaling factor should be built into the cost gradient. The cost gradient
-- function also must have the target data "baked in" (such as by partial
-- function application) so that it takes as its only argument the
-- activation from the last layer (network output).
iteration :: R -> (Matrix R -> Matrix R) -> [Layer] -> Matrix R  -> [Layer]
iteration eta dfCost network x = updated
    where fwdProp = runState . forwardPass $ network
          (acc, out) = fwdProp x
          backProp = runState . backwardPass eta $ acc
          (updated, _) = backProp $ dfCost out



{---------------- Network Initialization ----------------}
    
xavier :: Int -> Int -> NormalDistribution
xavier m n = normalDistr 0.0 (sqrt (1 / fromIntegral n))

uniInvRoot :: Int -> Int -> UniformDistribution
uniInvRoot m n = uniformDistr (-rs) rs
    where rs = 1 / (sqrt (fromIntegral n))

genWeights :: (ContGen d, PrimMonad m) =>
              Gen (PrimState m) -> Spec d -> m (Matrix R)
genWeights r spec = replicateM (_m * _n) (genContVar distr r)
                    >>= return . (_m >< _n)
    where _m = m spec
          _n = n spec
          distr = (winit spec) _m _n

-- Currently doesn't use a random generator
genBias :: (ContGen d, PrimMonad m) => Spec d -> m (Vector R)
genBias spec = return $ m spec |> repeat 0.0

constructLayer :: (ContGen d, PrimMonad m) =>
                  Gen (PrimState m) -> Spec d -> m Layer
constructLayer rgen spec = do
    _W   <- genWeights rgen spec
    _b   <- genBias spec
    return Layer { w = _W, b = _b,
                   actv = cmap $ fn spec,
                   dfActv = cmap $ dfn spec }
    
makeLayers :: (ContGen d, PrimMonad m) =>
              Gen (PrimState m) -> [Spec d] -> m [Layer]
makeLayers rgen = mapM (constructLayer rgen)



{---------------- Inference, Training, and Utilities ----------------}
          
inferBatch :: [Layer] -> Matrix R -> Matrix R
inferBatch network input = foldl
            (\ x l -> let _W = w l
                          _b = b l
                      in actv l $ (_W <> x)
                            + (repmat (asColumn _b) 1 (cols x)))
            input network

checkLayers :: [Spec d] -> Int
checkLayers list = foldl (\i (Spec m n _ _ _) -> if i > 0 && i == n
                                then m else 0) init list
    where init = n . head $ list

trainNetwork :: Int -> [(Matrix R, Matrix R)]
                -> [Layer] -> IO [Layer]
trainNetwork i batches network = runTraining
                    i 1 batches network

runTraining :: Int -> Int -> [(Matrix R, Matrix R)]
              -> [Layer] -> IO [Layer]
runTraining i c batches network =
    if c > i then
        return network
    else
        runEpochWithOutput c 0.1 -- (1 / (sqrt $ fromIntegral c))
                            network batches
        >>= runTraining i (c + 1) batches

runEpochWithOutput :: Int -> R -> [Layer]
                      -> [(Matrix R, Matrix R)]
                      -> IO [Layer]
runEpochWithOutput i eta network batches = do
    printf "====Epoch %d====\n" i
    result <- foldM applyEnum network (zip [1..] batches)
    printf "Network after epoch %d\n" i
    mapM_ printLayer (zip [1..] result)
    return result
    where applyEnum net (i, b) =
            runBatchWithOutput i eta net b

runBatchWithOutput :: Int -> R -> [Layer]
                      -> (Matrix R, Matrix R)
                      -> IO [Layer]
runBatchWithOutput i eta network (x, y) = do
    let result = iteration eta
          (dfQuadLoss y) network x
    let loss = quadLoss y (inferBatch result x)
    printf "batch %d loss: %f\n" i loss
    return result

-- Since the vector has only one element, that's
-- the one returned by maxElement
compareBinary :: Comparator
compareBinary res ans = ans == val
    where val = if (maxElement res) > 0.5
                then 1 else 0

compareByMax :: Comparator
compareByMax res ans = ans == val
    where val = assoc (size res) 0 [(maxIndex res, 1)]

-- Note that the lists of vectors must have the same length
processResults :: Metrics m => Comparator
                  -> [Vector R]
                  -> [Vector R]
                  -> [Vector R]
                  -> Bool -> Counter m -> m -> IO m
processResults _ [] [] [] _ _ metrics = return metrics
processResults comp (ex:examples) (res:results) (ans:answers)
               outputResults metricsCounter metrics = do
    let bool = comp res ans
    when outputResults (printResult bool ex res ans)
    processResults comp examples results answers
                   outputResults metricsCounter 
                   (metricsCounter comp res ans metrics)
    where newCount = 1 + count metrics
          printResult b e r a =
              let switchStr = 
                    if b then "matches answer"
                    else "does not match answer"
              in do
                    printf "Example %d: " newCount
                    print e
                    printf "result: "
                    print r
                    printf "%s: " switchStr
                    print a
                    putChar '\n'

testUsingComparator :: (Matrix R, Matrix R) -> Comparator
                        -> Bool -> [Layer] -> IO ()
testUsingComparator (test, ans) comp listOutputs network = do
    let results = inferBatch network test
    metrics <- processResults comp
                              (toColumns test)
                              (toColumns results)
                              (toColumns ans)
                              listOutputs
                              updateSimpleMetrics
                              (SimpleMetrics 0 0 0)
    let (total, c, w) = (count metrics,
                         correct metrics,
                         wrong metrics)
    printf "%d / %d correct\n" c total
    printf "%d / %d wrong\n" w total
    printf "%.3f%% accuracy\n" ((100 :: R) *
                                (fromIntegral c / fromIntegral total))

updateSimpleMetrics :: Comparator -> Vector R -> Vector R
                       -> SimpleMetrics -> SimpleMetrics
updateSimpleMetrics comp res ans mets =
    if ansCorrect then
        mets{simpleMetricsCount = newCount,
             correct = 1 + correct mets}
    else
        mets{simpleMetricsCount = newCount,
             wrong = 1 + wrong mets}
    where ansCorrect = comp res ans
          newCount = 1 + simpleMetricsCount mets

testBinaryClassifier :: (Matrix R, Matrix R) -> Bool
                        -> [Layer] -> IO ()
testBinaryClassifier (test, ans) listOutputs network = do
    let results = inferBatch network test
    BinaryMetrics total c_0 w_0 c_1 w_1
                    <- processResults compareBinary
                        (toColumns test)
                        (toColumns results)
                        (toColumns ans)
                        listOutputs
                        updateBinaryMetrics
                        (BinaryMetrics 0 0 0 0 0)
    printf "%d / %d true positive\n" c_1 (c_1 + w_0)
    printf "%d / %d false negative\n" w_0 (c_1 + w_0)
    printf "%d / %d true negative\n" c_0 (c_0 + w_1)
    printf "%d / %d false positive\n" w_1 (c_0 + w_1)
    printf "%.3f%% accuracy\n" ((100 :: R) *
                                (fromIntegral (c_1 + c_0)
                                    / fromIntegral total))

updateBinaryMetrics :: Comparator -> Vector R -> Vector R
                       -> BinaryMetrics -> BinaryMetrics
updateBinaryMetrics comp res ans mets =
    if correct then
        if binClass then
            mets{binaryMetricsCount = newCount,
                 correct_1 = 1 + correct_1 mets}
        else
            mets{binaryMetricsCount = newCount,
                 correct_0 = 1 + correct_0 mets}
    else
        if binClass then
            mets{binaryMetricsCount = newCount,
                 wrong_0 = 1 + wrong_0 mets}
        else
            mets{binaryMetricsCount = newCount,
                 wrong_1 = 1 + wrong_1 mets}
    where correct = comp res ans
          binClass = comp ans (scalar 1.0)
          newCount = 1 + binaryMetricsCount mets

shuffleSplit :: GenIO -> Int -> FFNN.Extractor a
                -> FFNN.Extractor a
                -> Vec.Vector a -> IO [(Matrix R, Matrix R)]
shuffleSplit gen n features target examples =
    Vec.thaw examples
    >>= shuffle gen (Vec.length examples) 
    >>= Vec.freeze
    >>= return . map matrify
               . map unzip
               . chunksOf n
               . Vec.foldr' extract []
    where extract ex pairs =
                (features ex, target ex)
                : pairs
          matrify (x, y) = (fromColumns x,
                            fromColumns y)

shuffle :: PrimMonad m => Gen (PrimState m)
           -> Int
           -> MVec.MVector (PrimState m) a
           -> m (MVec.MVector (PrimState m) a)
shuffle _ 0 vec = return vec
shuffle g i vec = do
        r <- uniformRM (0, i - 1) g
        MVec.swap vec r (i - 1)
        shuffle g (i - 1) vec

printLayer :: (Int, Layer) -> IO ()
printLayer (i, l) = do
    printf "Layer %d shape\n" i
    print . w $ l
    print . b $ l
    putChar '\n'

printBatch :: (Int, (Matrix R, Matrix R)) -> IO ()
printBatch (i, (x, y)) = do
    printf "Batch %d\n" i
    mapM_ print (toColumns x)
    putChar '\n'
