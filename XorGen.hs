module Main where

import Control.Monad
import Data.Bits
import Data.ByteString.Lazy
import Data.Csv
import Data.Vector (singleton)
import System.Environment
import System.Random.MWC

type Datum = (Double, Double, Double, Int, Int, Int, Int)

main :: IO ()
main = do
    args <- getArgs
    let (num : file : _) = args
    let n = read num
    rgen <- initialize (Data.Vector.singleton 94)
    ls <- replicateM n (genPoint rgen)
    -- mapM (print . show) ls
    -- return ()
    Data.ByteString.Lazy.writeFile file (encode ls)
    
genPoint :: GenIO -> IO Datum
genPoint g = do
    x <- uniformR (0.0, 1.0) g
    y <- uniformR (0.0, 1.0) g
    z <- uniformR (0.0, 1.0) g
    let (a, b, c, d) = label x y z
    return (x, y, z, a, b, c, d)

-- Labels are generated according to a point's position
-- in the unit cube, which is split up into octants.
-- The label is a vector of indicators usable for classifcation.
-- For the XOR dataset, octants that are diagonal from each
-- other (coordinates do not overlap along any direction)
-- will have the same label.
-- The point's label is calculated from the integer with bit
-- reprentation xyz, where each of x,y,z are 1 if the coordinates
-- of the point is > 0.5 and 0 otherwise. Integers whose least
-- three bits are binary negations of each other belong to the
-- same class.
label :: Double -> Double -> Double -> (Int, Int, Int, Int)
label x y z =
    case val of
        0 -> (1, 0, 0, 0)
        7 -> (1, 0, 0, 0)
        1 -> (0, 1, 0, 0)
        6 -> (0, 1, 0, 0)
        2 -> (0, 0, 1, 0)
        5 -> (0, 0, 1, 0)
        3 -> (0, 0, 0, 1)
        4 -> (0, 0, 0, 1)
    where bit a | a <= 0.5 = (0 :: Int)
                | otherwise = (1 :: Int)
          val = (shift (bit x) 2)
                .|. (shift (bit y) 1)
                .|. (bit z)