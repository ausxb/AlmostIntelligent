module FFNN (
        Layer, iteration, checkLayers,
        sigma, dfSigma, actv, dfActv
    ) where

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Data
import Control.Monad.State.Strict 

import Prelude (
        map, foldl, foldl1, uncurry, zip, head,
        Floating, Int, exp, fromIntegral, (+), (-),
        (/), (*), ($), (.), (==), (>), (<), (&&)
    )

data Layer = Layer {
    w :: Matrix R,
    b :: Vector R
}

type FwdAcc = [(Matrix R, Matrix R, Layer)]
type BackAcc = [Layer]

sigma :: Floating t => t -> t
sigma z = e/(e + 1)
    where e = exp z
          
dfSigma :: Floating t => t -> t
dfSigma x = c * (1 - c)
    where c = sigma x

-- Sigmoid activation, generalized so it applies to
-- matrices of input batches, as is dfActv.
actv :: (Floating t, Container c t) => c t -> c t
actv = cmap sigma

-- "Gradient" of vectorized sigmoid. Column vector form,
-- instead of true Jacobian. Use with Hadamard product.
dfActv :: (Floating t, Container c t) => c t -> c t
dfActv = cmap dfSigma

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
                    -- add _b to every column
                    let z = fromColumns $ map (+ _b)
                            $ toColumns (_W <> x)
                    in ((z, x, l) : acc, actv z) )
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
        \bp -> let dz = bp * dfActv z
                   _W = (-) (w l) $ cmap (* eta) $ stackOuter dz x
                   _b = (-) (b l) $ cmap (* eta) $ sumColumns dz
                   bp' = tr (w l) <> dz
               in (l{ w = _W, b = _b } : acc, bp') )

-- The reversed list from the forward pass is reversed again, yielding
-- an updated list of layers in the original order.
backwardPass :: R -> FwdAcc -> State (Matrix R) BackAcc
backwardPass eta = foldl (>>=) (state $ \x -> ([], x)) . map (updateLayer eta)

-- The learning rate is divided by the number of training examples here,
-- instead of in backwardPass, to simplify the implementation of backwardPass.
iteration :: Matrix R -> Matrix R -> R -> [Layer] -> [Layer]
iteration y x eta nn = updated
    where fwdProp = runState . forwardPass $ nn
          (acc, out) = fwdProp x
          backProp = runState .
                        backwardPass (eta / fromIntegral (cols out)) $ acc
          (updated, _) = backProp (y - out)
          
checkLayers :: [Layer] -> Int
checkLayers list = foldl (\i l -> if i > 0 && i == (cols (w l))
                       then rows (w l) else 0) init list
    where init = rows . w . head $ list
