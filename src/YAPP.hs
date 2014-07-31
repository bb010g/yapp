{-# LANGUAGE Trustworthy #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE RebindableSyntax, NoMonomorphismRestriction #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE ConstraintKinds #-}

module YAPP (
    -- * Standard types, classes and related functions

    -- ** Basic data types
    Bool(False, True),
    (&&), (||), not, otherwise,

    Maybe(Nothing, Just),
    maybe,

    Either(Left, Right),
    either,

    Ordering(LT, EQ, GT),
    Char, String,

    -- *** Tuples
    fst, snd, curry, uncurry,

    -- ** Basic type classes
    Eq((==), (/=)),
    Ord(compare, (<), (<=), (>=), (>), max, min),
    Enum(succ, pred, toEnum, fromEnum, enumFrom, enumFromThen,
         enumFromTo, enumFromThenTo),
    Bounded(minBound, maxBound),

    -- ** Numbers

    -- *** Numeric types
    Int, Integer, Float, Double,
    Rational,

    -- *** Numeric type classes
    Num((+), (-), (*), negate, abs, signum, fromInteger),
    Real(toRational),
    Integral(quot, rem, div, mod, quotRem, divMod, toInteger),
    Fractional((/), recip, fromRational),
    Floating(pi, exp, log, sqrt, (**), logBase, sin, cos, tan,
             asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh),
    RealFrac(properFraction, truncate, round, ceiling, floor),
    RealFloat(floatRadix, floatDigits, floatRange, decodeFloat,
              encodeFloat, exponent, significand, scaleFloat, isNaN,
              isInfinite, isDenormalized, isIEEE, isNegativeZero, atan2),

    -- *** Numeric functions
    subtract, even, odd, gcd, lcm, (^), (^^),
    fromIntegral, realToFrac,

    -- * Converting to and from @String@
    -- ** Converting to @String@
    ShowS,
    Show(showsPrec, showList, show),
    shows,
    showChar, showString, showParen,
    -- ** Converting from @String@
    ReadS,
    Read(readsPrec, readList),
    reads, readParen, read, lex,

    -- * Basic Input and output
    IO,
    -- ** Simple I\/O operations
    -- All I/O functions defined here are character oriented.  The
    -- treatment of the newline character will vary on different systems.
    -- For example, two characters of input, return and linefeed, may
    -- read as a single newline character.  These functions cannot be
    -- used portably for binary I/O.
    -- *** Output functions
    putChar,
    putStr, putStrLn, print,
    -- *** Input functions
    getChar,
    getLine, getContents, interact,
    -- *** Files
    FilePath,
    readFile, writeFile, appendFile, readIO, readLn,
    -- ** Exception handling in the I\/O monad
    IOError, ioError, userError,

    -- * Listy operations
    map, (++), filter,
    head, last, tail, init, null, length, (!!),
    reverse,
    -- ** Reducing lists (folds)
    foldl, foldl1, foldr, foldr1,
    -- *** Special folds
    and, or, any, all,
    sum, product,
    concat, concatMap,
    maximum, minimum,
    -- ** Building lists
    -- *** Scans
    scanl, scanl1, scanr, scanr1,
    -- *** Infinite lists
    iterate, repeat, replicate, cycle,
    -- ** Sublists
    take, drop, splitAt, takeWhile, dropWhile, span, break,
    -- ** Searching lists
    elem, notElem, lookup,
    -- ** Zipping and unzipping lists
    zip, zip3, zipWith, zipWith3, unzip, unzip3,
    -- ** Functions on strings
    lines, words, unlines, unwords,

    fold, foldMap, foldl', foldr', traverse, sequenceA, traverse_, sequenceA_, -- added
    headS, lastS, tailS, initS, length', iterate', iterateN, iterateN', -- added
    cycle', cycleN, cycleN', -- added

    -- ** Miscellaneous functions
    id, const, (.), flip, ($), until,
    asTypeOf, error, undefined,
    seq, ($!),
    (<<<), (>>>), ifThenElse, ifte, -- added

    -- ** Monads and functors and semigroupoids, oh my!
    -- changed severely
    Functor ((<$)), (<$>), (<$$>), ($>), void, -- map,
    Apply (), (<*>), (<**>), (<*), (*>),
    Alt (some, many), (<|>),
    Plus (), mplus, mzero,
    Applicative (), WrappedApplicative (WrapApplicative, unwrapApplicative),
    Alternative (), WrappedMonad (WrapMonad, unwrapMonad),
    Bind (join), (>>=), (=<<), (>>), (<<), (>=>), (<=<),
    Monad (fail), MonadPlus, returning, apDefault, guard, ap,
    mapM, mapM_, sequence, sequence_,

    Semigroupoid (), WrappedCategory (WrapCategory, unwrapCategory),
    Semi (Semi, getSemi),
    
    Semigroup ((<>), sconcat, times1p), timesN,
    WrappedMonoid (WrapMonoid, unwrapMonoid),
    Monoid (mempty), Option (Option, getOption), option,
    SMonoid,
) where

import GHC.Base
    ( Bool (False, True), (&&), (||), not, otherwise
    , Ordering (LT, EQ, GT)
    , Char, String
    , Eq ((==), (/=))
    , Ord (compare, (<), (<=), (>=), (>), max, min)
    , Int
    , const, flip, ($), until, asTypeOf
    , error, undefined
    , seq
    )
import Text.Read ()
import GHC.Enum (Enum (succ, pred, toEnum, fromEnum, enumFrom,
                       enumFromThen, enumFromTo, enumFromThenTo)
                ,Bounded (minBound, maxBound)
                )
import GHC.Num
    ( Integer, Num ((+), (-), (*), negate, abs, signum, fromInteger)
    , subtract
    )
import GHC.Real
    ( Rational, Real (toRational)
    , Integral (quot, rem, div, mod, quotRem, divMod, toInteger)
    , Fractional ((/), recip, fromRational)
    , RealFrac (properFraction, truncate, round, ceiling, floor)
    , even, odd, gcd, lcm, (^), (^^), fromIntegral, realToFrac
    )
import GHC.Float
    ( Float, Double
    , Floating
        ( pi, exp, log, sqrt, (**), logBase, sin, cos, tan, asin, acos, atan
        , sinh, cosh, tanh, asinh, acosh, atanh
        )
    , RealFloat
        ( floatRadix, floatDigits, floatRange, decodeFloat, encodeFloat
        , exponent, significand, scaleFloat, isNaN, isInfinite, isDenormalized
        , isIEEE, isNegativeZero, atan2
        )
    )
import GHC.Show
    ( ShowS, Show (showsPrec, showList, show)
    , shows, showChar, showString, showParen
    ) 
import Text.Read
    ( ReadS, Read (readsPrec, readList)
    , reads, readParen, read, lex
    )

import Data.Either (Either (Left, Right), either)
import Data.Maybe (Maybe (Nothing, Just), maybe)
import Data.Tuple (fst, snd, curry, uncurry)

import System.IO
    ( IO, putChar, putStr, putStrLn, print
    , getChar, getLine, getContents, interact
    , FilePath, readFile, writeFile, appendFile, readIO, readLn
    )
import System.IO.Error (IOError, ioError, userError)

import Prelude (($!))

import Data.Foldable
    ( Foldable, Foldable (fold, foldMap, foldr, foldr', foldl, foldl'
    , foldr1, foldl1), mapM_, sequence_, and, or, any, all, sum, product
    , concat, concatMap, maximum, minimum, elem, notElem
    , traverse_, sequenceA_
    )
import Data.Traversable
    ( traverse, sequenceA
    , mapM,     sequence
    )
import Data.List
    ( filter, tail, init, (!!), reverse
    , scanl, scanl1, scanr, scanr1
    , take, drop, splitAt, takeWhile, dropWhile, span
    , break, lookup, zip, zip3, zipWith, zipWith3, unzip, unzip3, lines, words
    )

import qualified Data.Semigroupoid (Semigroupoid (o))
import Data.Semigroupoid (Semigroupoid, WrappedCategory,
                          WrappedCategory (WrapCategory, unwrapCategory),
                          Semi, Semi (Semi, getSemi))
import Control.Category (Category (id))

import Data.Semigroup
    ( Semigroup, Semigroup ((<>), sconcat, times1p), timesN, WrappedMonoid
    , WrappedMonoid (WrapMonoid, unwrapMonoid), Monoid
    , Monoid (mempty), Option, Option (Option, getOption)
    , option
    )

import qualified Data.Functor (Functor (fmap))
import           Data.Functor (Functor, Functor ((<$)), ($>), (<$>), void)
import qualified Data.Functor.Apply (Apply ((<.>), (.>), (<.)), (<..>))
import           Data.Functor.Apply (Apply, WrappedApplicative,
                                     WrappedApplicative (WrapApplicative,
                                                         unwrapApplicative))
import qualified Data.Functor.Alt (Alt ((<!>)))
import           Data.Functor.Alt (Alt, Alt (some, many))
import qualified Data.Functor.Plus (Plus (zero))
import           Data.Functor.Plus (Plus)
import qualified Control.Applicative (Applicative (pure))
import           Control.Applicative (Applicative, Alternative,
                                      WrappedMonad, WrappedMonad
                                      (WrapMonad, unwrapMonad))
import qualified Data.Functor.Bind (Bind ((>>-)), (-<<), (-<-), (->-))
import           Data.Functor.Bind (Bind, Bind (join), returning, apDefault)
import Control.Monad (Monad, Monad (fail), MonadPlus)

ifThenElse :: Bool -> a -> a -> a
ifThenElse p t f = case p of
    True  -> t
    False -> f

ifte :: Bool -> a -> a -> a
ifte = ifThenElse

infixr 9 .
(.) :: Semigroupoid c => c j k -> c i j -> c i k
(.) = Data.Semigroupoid.o

infixr 1 >>>, <<<
(<<<) :: Semigroupoid c => c j k -> c i j -> c i k
(<<<) = (.)
(>>>) :: Semigroupoid c => c i j -> c j k -> c i k
(>>>) = flip (.)

type SMonoid a = (Semigroup a, Monoid a)

infixr 5 ++
(++) :: Semigroup s => s -> s -> s
(++) = (<>)

map :: Functor f => (a -> b) -> f a -> f b
map = Data.Functor.fmap
infixl 4 <$$>
(<$$>) :: Functor f => f a -> (a -> b) -> f b
(<$$>) = flip (<$>)
infixl 4 <*>, <*, *>, <**>
(<*>) :: Apply f => f (a -> b) -> f a -> f b
(<*>) = (Data.Functor.Apply.<.>)
(<*) :: Apply f => f a -> f b -> f a
(<*) = (Data.Functor.Apply.<.)
(*>) :: Apply f => f a -> f b -> f b
(*>) = (Data.Functor.Apply..>)
(<**>) :: Apply f => f a -> f (a -> b) -> f b
(<**>) = (Data.Functor.Apply.<..>)
infixl 3 <|>
(<|>) :: Alt f => f a -> f a -> f a
(<|>) = (Data.Functor.Alt.<!>)
mplus :: Alt f => f a -> f a -> f a
mplus = (<|>)
mzero :: Plus f => f a
mzero = Data.Functor.Plus.zero
infixl 1 >>, >>=, <<, =<<
(>>=) :: Bind f => f a -> (a -> f b) -> f b
(>>=) = (Data.Functor.Bind.>>-)
(=<<) :: Bind f => (a -> f b) -> f a -> f b
(=<<) = (Data.Functor.Bind.-<<)
(<<) :: Apply f => f a -> f b -> f a
(<<) = (<*)
(>>) :: Apply f => f a -> f b -> f b
(>>) = (*>)
infixr 1 <=<, >=>
(<=<) :: Bind f => (b -> f c) -> (a -> f b) -> (a -> f c)
(<=<) = (Data.Functor.Bind.-<-)
(>=>) :: Bind f => (a -> f b) -> (b -> f c) -> (a -> f c)
(>=>) = (Data.Functor.Bind.->-)
return :: Applicative f => a -> f a
return = Control.Applicative.pure
guard :: (Alternative f, Plus f) => Bool -> f ()
guard p = if p then return () else mzero
ap :: Apply f => f (a -> b) -> f a -> f b
ap = (<*>)

head :: Foldable t => t a -> a
head = foldr const (errorEmptyFoldable "head")
headS :: Foldable t => t a -> Maybe a
headS = foldr (const . Just) Nothing
last :: Foldable t => t a -> a
last = foldl' (flip const) (errorEmptyFoldable "last")
lastS :: Foldable t => t a -> Maybe a
lastS = foldl' (const Just) Nothing
tailS :: [a] -> Maybe [a]
tailS [] = Nothing
tailS xs = Just $ tail xs
{-# INLINE tailS #-}
initS :: [a] -> Maybe [a]
initS [] = Nothing
initS xs = Just $ init xs
{-# INLINE initS #-}
null :: Foldable t => t a -> Bool
null (headS -> Nothing) = True
null _ = False
length :: Foldable t => t a -> Integer
length = foldr (const (+1)) 0
length' :: (Num n, Foldable t) => t a -> n
length' = foldr (const (+1)) 0

iterate :: (Semigroup (t (a -> a)), Applicative t) => (a -> a) -> a -> t a
iterate f x = ($ x) <$> iterate' f id
iterate' :: (Semigroupoid c, Semigroup (t (c a b)), Applicative t) =>
            c b b -> c a b -> t (c a b)
iterate' f x = return x ++ iterate' f (f . x)
iterateN :: (SMonoid (t (a -> a)), Applicative t) =>
            (a -> a) -> Integer -> a -> t a
iterateN f n x = ($ x) <$> iterateN' f n id
iterateN' :: (Semigroupoid c, SMonoid (t (c a b)), Applicative t) =>
             c b b -> Integer -> c a b -> t (c a b)
iterateN' _ 0 _ = mempty
iterateN' f n x = return x ++ iterateN' f (n - 1) (f . x)

repeat :: (Semigroup (t a), Applicative t) => a -> t a
repeat x = return x ++ repeat x
replicate :: (SMonoid (t a), Applicative t) =>
             Integer -> a -> t a
replicate 0 _ = mempty
replicate n x = return x ++ replicate (n - 1) x

cycle :: (Semigroup (t a), Foldable t) => t a -> t a
cycle (null -> True) = errorEmptyFoldable "cycle"
cycle tx = tx ++ cycle tx
cycle' :: (Semigroup (t a), Foldable t) => t a -> t a
cycle' tx@(null -> True) = tx
cycle' tx = tx ++ cycle' tx
cycleN :: (SMonoid (t a), Foldable t) => Integer -> t a -> t a
cycleN 0 _ = mempty
cycleN _ (null -> True) = errorEmptyFoldable "cycleN"
cycleN n tx = tx ++ cycleN (n - 1) tx
cycleN' :: (SMonoid (t a), Foldable t) => Integer -> t a -> t a
cycleN' 0 _ = mempty
cycleN' _ tx@(null -> True) = tx
cycleN' n tx = tx ++ cycleN' (n - 1) tx

errorEmptyFoldable :: String -> a
errorEmptyFoldable f = error $ "YAPP."++f++": empty Foldable"

unlines :: Foldable t => t String -> String
unlines = concatMap (++"\n")
unwords :: Foldable t => t String -> String
unwords (null -> True) = ""
unwords ws = foldr1 (\w s -> w ++ ' ':s) ws

-- From the normal Prelude

#ifdef __HADDOCK__
-- | The value of @'seq' a b@ is bottom if @a@ is bottom, and otherwise
-- equal to @b@.  'seq' is usually introduced to improve performance by
-- avoiding unneeded laziness.
seq :: a -> b -> b
seq _ y = y
#endif
