module FFNN (
        Layer(..), Spec(..),
        iteration, checkLayers, inferBatch,
        sigma, dfSigma, tanh, dfTanh,
        xavier, makeLayers
    ) where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Control.Monad
import Control.Monad.Primitive
import Control.Monad.State.Strict

import System.Random.MWC
import Statistics.Distribution
import Statistics.Distribution.Normal

import Prelude (
        IO, Floating, Int,
        map, foldl, foldl1, mapM, zip, head, repeat,
        uncurry, snd, exp, fromIntegral, sqrt,
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

type FwdAcc = [(Matrix R, Matrix R, Layer)]
type BackAcc = [Layer]

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

sumColumns :: Matrix R -> Vector R
sumColumns = foldl1 (+) . toColumns

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
-- scaling factor should be precomputed into the learning rate or 
-- be part of the cost gradient. The cost gradient function should have
-- the target data "baked in" so that it takes as its only argument the
-- activation from the last layer (network output).
iteration :: R -> (Matrix R -> Matrix R) -> [Layer] -> Matrix R  -> [Layer]
iteration eta dfCost network x = updated
    where fwdProp = runState . forwardPass $ network
          (acc, out) = fwdProp x
          backProp = runState . backwardPass eta $ acc
          (updated, _) = backProp $ dfCost out
          
-- inferSingle :: Vector R -> [Layer] -> Vector R
          
inferBatch :: Matrix R -> [Layer] -> Matrix R
inferBatch = foldl (\ x l -> let _W = w l
                                 _b = b l
                             in actv l $ fromColumns $ map (+ _b)
                                $ toColumns $ _W <> x)
          
checkLayers :: [Spec d] -> Int
checkLayers list = foldl (\i (Spec m n _ _ _) -> if i > 0 && i == n
                                then m else 0) init list
    where init = n . head $ list

xavier :: Int -> Int -> NormalDistribution
xavier m n = normalDistr 0.0 (sqrt (1 / fromIntegral n))

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
