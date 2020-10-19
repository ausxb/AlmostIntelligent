module Main where

import Control.Monad
import Data.Bits
import Data.ByteString.Lazy
import Data.Csv
import System.Environment
import System.Random.MWC

type Datum = (Double, Double, Int)

-- The command line arguments m1, b1, m2, and b2
-- define two lines using slope intercept form.
-- Generated points that lie below y = m1 x + b1
-- and above y = m2 x + b are labeled as 1
-- and any point outside that region is labeled 0.
main :: IO ()
main = do
    args <- getArgs
    let (num : file : m1 : b1 : m2 : b2 : []) = args
    let n = read num
    rgen <- createSystemRandom
    let below = belowLine (read m1) (read b1)
    let above = aboveLine (read m2) (read b2)
    let isInOverlap = \ x y ->
            if (below x y) && (above x y) then 1 else 0
    ls <- replicateM n (genPoint rgen isInOverlap)
    Data.ByteString.Lazy.writeFile file (encode ls)

genPoint :: GenIO -> (Double -> Double -> Int) -> IO Datum
genPoint g label = do
    x <- uniformR (-10.0, 10.0) g
    y <- uniformR (-10.0, 10.0) g
    return (x, y, label x y)

-- Takes slope-intercept parameters of a line and generates
-- a function that returns true if a point is above the line
aboveLine :: Double -> Double -> (Double -> Double -> Bool)
aboveLine m b = \ x y -> ((-m) * x + y) > b

-- Takes slope-intercept parameters of a line and generates
-- a function that returns true if a point is below the line
belowLine :: Double -> Double -> (Double -> Double -> Bool)
belowLine m b = \ x y -> ((-m) * x + y) < b
