module Main where

import FFNN

import Control.Monad
import Data.ByteString.Lazy (readFile)
import Data.Csv
import qualified Data.Vector as Vec
import Numeric.LinearAlgebra
import System.Environment
import System.Random.MWC

type Datum = (Double, Double, Double, Double,
                Double, Double, Double)

xor_spec = [Spec { m = 9, n = 3,
                   fn = sigma,
                   dfn = dfSigma,
                   winit = uniInvRoot },
            Spec { m = 9, n = 9,
                   fn = sigma,
                   dfn = dfSigma,
                   winit = uniInvRoot },
            Spec { m = 4, n = 9,
                   fn = sigma,
                   dfn = dfSigma,
                   winit = uniInvRoot }]

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
    batches <- shuffleSplit gen 1000
                datumFeatures datumTarget train
    
    -- mapM_ printBatch (zip [1..] batches)
    trainNetwork 1 batches network
    
    test_batch <- shuffleSplit gen 1000
                    datumFeatures datumTarget test
    testUsingComparator (head test_batch) compareByMax
                        False network
    
    return ()

datumFeatures :: Datum -> Vector R
datumFeatures (x,y,z,a,b,c,d) = fromList [x,y,z]

datumTarget :: Datum -> Vector R
datumTarget (x,y,z,a,b,c,d) = fromList [a,b,c,d]

loadCsv :: String -> IO (Vec.Vector Datum)
loadCsv file = do
    csv <- Data.ByteString.Lazy.readFile file
    case decode NoHeader csv of
        Left msg -> putStr msg >> return Vec.empty
        Right vec -> return vec
