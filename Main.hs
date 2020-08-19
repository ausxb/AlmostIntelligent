module Main where

import FFNN
import System.Random.MWC
import Control.Monad
import Data.Vector (singleton)
import Data.Word (Word32)

-- Testing random initialization
main :: IO ()
main = do
    let spec = [Spec { m = 10, n = 10,
                       fn = FFNN.tanh,
                       dfn = dfTanh,
                       winit = xavier },
                Spec { m = 5, n = 10,
                       fn = FFNN.tanh,
                       dfn = dfTanh,
                       winit = xavier }]
    let badDims = checkLayers spec == 0
    when badDims (putStrLn
        "Mismatch in layer dimensions")
    guard $ not badDims
    -- gen <- initialize (singleton 94)
    gen <- createSystemRandom
    net <- makeLayers gen spec
    mapM printLayer (zip [1..] net)
    return ()
    
printLayer :: (Int, Layer) -> IO ()
printLayer (i, l) = do
    putStr $ "Layer " ++ show i ++ " shape "
    print . w $ l
    putChar '\n'
