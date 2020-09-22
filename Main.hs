module Main where

import FFNN

import Control.Monad
import Control.Monad.Primitive
import Data.ByteString.Lazy (readFile)
import Data.Csv
import Data.List.Split (chunksOf)
import Data.STRef
import Data.Vector as Vec
        (Vector, empty, toList, splitAt, length, 
         freeze, thaw)
import Data.Vector.Mutable as MVec
        (MVector, STVector, swap)
import Numeric.LinearAlgebra as LinAlg
import Numeric.LinearAlgebra.Data as LinAlg.Data
import System.Environment
import System.Random.MWC

type Datum = (Double, Double, Double, Double,
                Double, Double, Double)

-- type Features = (x -> LinAlg.Vector R)

xor_spec = [Spec { m = 4, n = 3,
                   fn = FFNN.tanh,
                   dfn = dfTanh,
                   winit = xavier },
            Spec { m = 4, n = 4,
                   fn = FFNN.tanh,
                   dfn = dfTanh,
                   winit = xavier }]

main :: IO ()
main = do
    args <- getArgs
    let (file : _) = args
    
    data_list <- loadCsv file
    
    when (null data_list) (putStrLn $
        "Couldn't load " ++ file ++ ", wrong format?")
    guard $ not (null data_list)
    
    let dimMismatch = checkLayers xor_spec == 0
    when dimMismatch (putStrLn
        "Mismatch in layer dimensions")
    guard $ not dimMismatch
    
    gen <- createSystemRandom
    network <- makeLayers gen xor_spec
    mapM_ printLayer (zip [1..] network)
    
    let (train, test) = Vec.splitAt 10000 data_list
    batches <- shuffleSplit gen 1000 train
    
    -- runEpochWithOutput 1 0.085 network batches
    run 500 500 batches network
    
    return ()

run :: Int -> Int -> [(Matrix R, Matrix R)]
       -> [Layer] -> IO [Layer]
run i 0 batches network = return network
run i c batches network = runEpochWithOutput
    (i - c + 1) (1 / (sqrt $ fromIntegral i))
    network batches
    >>= run i (c - 1) batches

runEpochWithOutput :: Int -> Double -> [Layer]
                      -> [(Matrix R, Matrix R)]
                      -> IO [Layer]
runEpochWithOutput i eta network batches = do
    putStrLn $ "====Epoch " ++ show i ++ "===="
    result <- foldM applyEnum network (zip [1..] batches)
    putStrLn $ "Network after epoch " ++ show i
    mapM_ printLayer (zip [1..] result)
    return result
    where applyEnum net (i, b) =
            runBatchWithOutput i eta net b
    
runBatchWithOutput :: Int -> Double -> [Layer]
                      -> (Matrix R, Matrix R)
                      -> IO [Layer]
runBatchWithOutput i eta network (x, y) = do
    let result = iteration (eta /
          (fromIntegral . LinAlg.cols $ x))
          (\a -> a - y) network x
    let loss = cost y (inferBatch x result)
    putStrLn $ "batch " ++ show i ++ " loss: " ++ show loss
    return result

cost :: Matrix R -> Matrix R -> Double
cost y x = (sum (map normsq diffs)) / (2 * n)
    where diffs = (LinAlg.Data.toColumns (y - x))
          normsq z = LinAlg.dot z z
          n = fromIntegral $ LinAlg.Data.cols x

shuffleSplit :: GenIO -> Int -> Vec.Vector Datum
                -> IO [(Matrix R, Matrix R)]
shuffleSplit gen n examples = do
    Vec.thaw examples
        >>= shuffle gen (Vec.length examples) 
        >>= Vec.freeze
        >>= \ ex -> return $
            map makeBatch $ chunksOf n $ Vec.toList ex
    where separate (x,y,z,a,b,c,d) =
                (LinAlg.Data.fromList [x,y,z],
                 LinAlg.Data.fromList  [a,b,c,d])
          matrify (x, y) = (LinAlg.Data.fromColumns x,
                            LinAlg.Data.fromColumns y)
          makeBatch = matrify . unzip . map separate
          
shuffle :: PrimMonad m => Gen (PrimState m)
                          -> Int
                          -> MVec.MVector (PrimState m) a
                          -> m (MVec.STVector (PrimState m) a)
shuffle _ 0 vec = return vec
shuffle g i vec = do
        r <- uniformRM (0, i - 1) g
        MVec.swap vec r (i - 1)
        shuffle g (i - 1) vec
                            
loadCsv :: String -> IO (Vec.Vector Datum)
loadCsv file = do
    csv <- Data.ByteString.Lazy.readFile file
    case decode NoHeader csv of
        Left msg -> putStr msg >> return empty
        Right vec -> return vec

printLayer :: (Int, Layer) -> IO ()
printLayer (i, l) = do
    putStr $ "Layer " ++ show i ++ " shape "
    print . w $ l
    putChar '\n'
