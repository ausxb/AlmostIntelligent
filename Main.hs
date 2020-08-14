module Main where

import FFNN
import System.Random.MWC
import Data.Vector (singleton)
import Data.Word (Word32)

-- Testing random initialization
main :: IO ()
main = do
    let arch = [(10, 10), (5, 10)]
    --gen <- (initialize (singleton 94))
    gen <- createSystemRandom
    net <- initWeights xavier gen arch
    mapM printLayer (zip [1..] net)
    return ()
    
printLayer :: (Int, Layer) -> IO ()
printLayer (i, l) = do
    putStr $ "Layer " ++ show i ++ " shape "
    print . w $ l
    putChar '\n'
