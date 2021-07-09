#!/usr/local/bin/python

# <h4>Introduction</h4>

# This document explains the inner workings of a program that
# demodulates Franke and Taylor's FT8 digital mode. I hope the
# explanation helps others write their own home-brew FT8 software,
# perhaps using this code as a starting point. You can find the
# program's Python source code at
# <a href="https://github.com/rtmrtmrtmrtm/basicft8">https://github.com/rtmrtmrtmrtm/basicft8</a>.

# <h4>FT8 Summary</h4>

# An FT8 cycle starts every 15 seconds, at 0, 15, 30 and 45 seconds
# past the minute. An FT8 signal starts 0.5 seconds into a cycle and
# lasts 12.64 seconds. It consists of 79 symbols, each 0.16 seconds
# long. Each symbol is a single steady tone. For any given signal
# there are eight possible tones (i.e. it is 8-FSK). The tone spacing
# is 6.25 Hertz.

# To help receivers detect the presence of signals and to estimate
# where they start in time and in frequency, there are three sequences
# of seven fixed tones embedded in each signal. Each fixed sequence is
# called a Costas synchronization array, and consists of the tone
# sequence 2, 5, 6, 0, 4, 1, and 3.

# The other 58 symbols carry information.
# Each symbol conveys 3 bits (since it's 8-FSK),
# yielding 174 bits. The 174 bits are a "codeword", which must be
# given to a Low Density Parity Check (LDPC) decoder to yield 87 bits.
# The LDPC decoder uses the extra bits to correct bits corrupted by
# noise, interference, fading, etc. The 87 bits consists of 75 bits of
# "packed message" plus a 12-bit Cyclic Redundancy Check (CRC). The
# CRC is an extra check to verify that the output of the LDPC decoder
# really is a proper message.

# The 75-bit packed message can have one of a few formats. The most
# common format is two call signs (each packed into 28 bits) and
# either a signal strength report or a grid square (packed into 16
# bits).

# This demodulator uses upper-side-band audio from a receiver, so it
# sees a roughly 2500-Hz slice of spectrum. An FT8 signal is 50 Hz
# wide (8 tones times 6.25 Hz per tone), and there may be many signals
# in the audio. The demodulator does not initially know the
# frequencies of any of them, so it must search in the 2500 Hz. While
# most signals start roughly 0.5 seconds into a cycle, differences in clock
# settings mean that the demodulator must search in time as well as
# frequency.

# To illustrate, here is a spectrogram from an FT8 cycle. Time
# progresses along the x-axis, starting at the beginning of a cycle,
# and ending about 13.5 seconds later. The y-axis shows a slice of
# about 800 Hz. Three signals are visible; the middle signal starts a
# little early. Each signal visibly shifts frequency as it progresses
# from one 8-FSK symbol to the next. With a bit of imagination you can
# see that the signals have identical first and last runs of seven
# symbols; these are the Costas arrays.

# <img src="ft8example.jpg">

# Here's a summary of the stages in which an FT8 sender constructs a signal,
# along with the size of each stage's output.
# <ul>
# <li> Start with a message like CQ AB1HL FN42.
# <li> Pack the message: 72 bits.
# <li> Add 3 zero bits: 75 bits.
# <li> Add a CRC-12: 87 bits.
# <li> Encode with LDPC: 174 bits.
# <li> Turn each 3 bits into a symbol number from 0 to 8: 58 symbol numbers.
# <li> Add three 7-symbol Costas arrays: 79 symbol numbers.
# <li> Generate 8-FSK audio: 12.64 seconds of audio.
# <li> Send the audio to a radio transmitter.
# </ul>

# <h4>Demodulator Strategy</h4>

# This demodulator looks at an entire FT8 cycle's worth of audio
# samples at a time. It views the audio as a two-dimensional matrix of
# "bins", with frequency in 6.25-Hz units along one axis, and time in
# 0.16-second symbol-times along the other axis. Much like the
# spectrogram image above.  Each bin corresponds to a
# single tone lasting for one symbol time. Each bin's value indicates
# how much signal energy was received at the corresponding frequency
# and time. This arrangement is convenient because, roughly speaking,
# one can demodulate 8-FSK by seeing which of the relevant 8 bins is
# strongest during each symbol time. The demodulator searches for
# Costas synchronization arrays in this matrix. For each
# plausible-looking triplet of Costas arrays at the same base
# frequency and with the right spacing in time, the demodulator
# extracts bits from FSK symbols and sees if the LDPC decoder can
# interpret the bits as a correct codeword. If the LDPC succeeds, and the
# CRC is correct, the demodulator unpacks the message and prints it.

# The demodulator requires an audio sample rate of 12000
# samples/second. It turns the audio into bins by
# repeated use of Fast Fourier Transforms (FFTs), one per symbol time.
# A symbol time is 1920 samples. Each FFT takes 1920 audio samples and
# returns (1920/2)+1 output bins, each containing the strength of a
# different frequency within those audio samples. The FFT output bins
# correspond to frequencies at multiples of 12000/1920, or 6.25 Hz,
# which is the FT8 tone spacing. Thus the demodulator forms its matrix
# of bins by handing the audio samples, 1920 at a time, to FFTs, and
# stacking the results to form a matrix.

# This program catches only a fraction of the FT8 signals that wsjt-x
# can decode. Perhaps the most serious deficiency is that the program
# only works well for signals that arrive aligned near 1920-sample
# boundaries in time, and near 6.25-Hz boundaries in frequency. It
# would be more clever to look for signals on half- or quarter-bin
# boundaries, in time and in frequency.

# <h4>Code</h4>

# With preliminaries out of the way, here is the demodulator code.

##
## simple FT8 decoder
##
## Robert Morris, AB1HL
##

# These imports tell python which modules to include. Numpy provides
# FFTs and convenient array manipulation. pyaudio provides access to
# sound cards.

import numpy
import pyaudio
import wave
import sys
import time
import re
import threading

## FT8 modulation and protocol definitions.
## 1920-point FFT at 12000 samples/second
##   yields 6.25 Hz spacing, 0.16 seconds/symbol
## encode chain:
##   75 bits
##   append 12 bits CRC (for 87 bits)
##   LDPC(174,87) yields 174 bits
##   that's 58 3-bit FSK-8 symbols
##   insert three 7-symbol Costas sync arrays
##     at symbol #s 0, 36, 72 of final signal
##   thus: 79 FSK-8 symbols
## total transmission time is 12.64 seconds

class FT8:

    # The process() function demodulates one FT8 cycle. It is
    # called with 13.64 seconds of audio at 12,000
    # samples/second: samples is a numpy array with about 164,000
    # elements. 13.64 seconds is enough samples for a whole signal
    # (12.64 seconds) plus half a second of slop at the start and end.
    # process() computes for at most ten seconds so that it doesn't overlap
    # with the call for the next 15-second FT8 cycle; the call to
    # time.time() records the starting wall-clock time in seconds.

    def process(self, samples):
        ## set up to quit after 10 seconds.
        t0 = time.time()

        # How many symbols does samples hold? // is Python's integer division.
        # self.block is 1920, the number of samples in one FT8 symbol.

        nblocks = len(samples) // self.block ## number of symbol times in samples[]

        # Perform one FFT for each symbol-time's worth of samples.
        # Each FFT returns an array with nbins elements. The matrix m
        # will hold the results; m[i][j] holds the strength of
        # frequency j*6.25 Hz during the i'th symbol-time.
        # The FFT returns complex numbers that indicate
        # phase as well as amplitude; the abs() essentially throws away
        # the phase.

        ## one FFT per symbol time.
        ## each FFT bin corresponds to one FSK tone.
        nbins = (self.block // 2) + 1        ## number of bins in FFT output
        m = numpy.zeros((nblocks, nbins))
        for i in range(0, nblocks):
            block = samples[i*self.block:(i+1)*self.block]
            bins = numpy.fft.rfft(block)
            bins = abs(bins)
            m[i] = bins

        # Much of this code deals with arrays of numbers. Thus block
        # above holds the 1920 samples of a single symbol time, bins
        # holds the (1920/2)+1 FFT result elements, and the final
        # assignment copies bins to a slice through the m matrix.

        # Next the code will look for Costas arrays in m. A Costas
        # array in m looks like a 7x8 sub-matrix with one
        # high-valued element in each column, and the other elements
        # with low values. The high-valued elements correspond to the
        # i'th tone of the Costas array. We'll find Costas arrays in m
        # using exhaustive matrix multiplications of sub-matrices of m
        # with a Costas template matrix that has a 1 in each element
        # that should be high-valued, and a -1/7 in each other
        # element. The sum over the product's elements will be large
        # if we've found a real Costas sync array, and close to zero
        # otherwise. The reason to use -1/7 rather than -1 is to avoid
        # having the results dominated by the sum of the large number
        # of elements that should be low-valued.

        ## prepare a template of the Costas sync array
        ## we expect to receive at start, middle, and
        ## end of a transmission.
        costas_matrix = numpy.ones((7, 8)) * (-1 / 7.0)
        costas_symbols = [ 2, 5, 6, 0, 4, 1, 3 ]
        for i in range(0, len(costas_symbols)):
            costas_matrix[i][costas_symbols[i]] = 1

        # Now examine every symbol-time and FFT frequency bin at which
        # a signal could start (there are a few thousand of them). The
        # signal variable holds the 79x8 matrix for one signal. Sum
        # the strengths of the Costas arrays for that potential
        # signal, and append the strength to the candidates array.
        # candidates will end up holding the likelihood of there being
        # an FT8 signal for every possible starting position.

        ## first pass: look for Costas sync arrays.
        candidates = [ ]
        ## for each start time
        for bi in range(0, nbins-8):
            ## a signal's worth of FFT bins -- 79 symbols, 8 FSK tones.
            for si in range(0, nblocks - 79):
                signal = m[si:si+79,bi:bi+8]
                strength = 0.0
                strength += numpy.sum(signal[0:7,0:8] * costas_matrix)
                strength += numpy.sum(signal[36:43,0:8] * costas_matrix)
                strength += numpy.sum(signal[72:79,0:8] * costas_matrix)
                candidates.append( [ bi, si, strength ] )

        # Sort the candidate signals, strongest first.

        ## sort the candidates, strongest Costas sync first.
        candidates = sorted(candidates, key = lambda e : -e[2])

        # Now we'll look at the candidate start positions, strongest
        # first, and see if the LDPC decoder can extract a signal from
        # each of them. This is the second pass in a two-pass scheme:
        # the first pass is the code above that looks
        # for plausible Costas sync arrays, and the second pass is the
        # code below that tries LDPC
        # decoding on the strongest candidates. Why two
        # passes, rather than simply trying LDPC decoding at
        # each possible signal start position? Because LDPC decoding
        # takes a lot of CPU time, and in our 10-second budget there's
        # only enough time to try it on a modest number of candidates.
        # The Costas sync detection above, however, is cheap enough
        # that it's no problem to do it for every possible signal
        # start position. The result is that we only try expensive
        # LDPC decoding for start positions that have a good chance of
        # actually being signals.

        # The assignment to signal extracts the 79x8 bins that
        # correspond to this candidate signal. signal[3][4] contains the
        # strength of the 4th FSK tone in symbol 3. If it is the
        # highest among the 8 elements of signal[3], then
        # symbol 3's value is probably 4 (yielding the three bits 100).
        # The call to process1() does most of the remaining work (see
        # below). This loop quits after 10 seconds.

        ## look at candidates, best first.
        for cc in candidates:
            if time.time() - t0 >= 10:
                ## quit after 10 seconds.
                break

            bi = cc[0]
            si = cc[1]
            ## a signal's worth of FFT bins -- 79 symbols, 8 FSK tones.
            signal = m[si:si+79,bi:bi+8]

            msg = self.process1(signal)

            if msg != None:
                bin_hz = self.rate / float(self.block)
                hz = bi * bin_hz
                print("%6.1f %s" % (hz, msg))


    # fsk_bits() is a helper function that turns a 58x8 array of tone
    # strengths into 58*3 bits. It does this by deciding which tone is
    # strongest for each of the 58 symbols. s58 holds, for each symbol, the index of the
    # strongest tone at that symbol time. bits3 generates each symbol's three bits, and
    # numpy.concatenate() flattens them into 174 bits.

    ## given 58 symbols worth of 8-FSK tones,
    ## decide which tone is the real one, and
    ## turn the resulting symbols into bits.
    ## returns log-likelihood for each bit,
    ## since that's what the LDPC decoder wants,
    ## but the probability is faked.
    def fsk_bits(self, m58):
        ## strongest tone for each symbol time.
        s58 = [ numpy.argmax(x) for x in m58 ]

        ## turn each 3-bit symbol into three bits.
        ## most-significant bit first.
        bits3 = [ [ (x>>2)&1, (x>>1)&1, x&1 ] for x in s58 ]
        a174 = numpy.concatenate(bits3)

        return a174

    # process() calls process1() for each candidate signal.
    # m79[0..79][0..8] holds the eight tone strengths for each received
    # symbol.

    ## m79 is 79 8-bucket mini FFTs, for 8-FSK demodulation.
    ## m79[0..79][0..8]
    ## returns None or a text message.
    def process1(self, m79):

        # Drop the three 7-symbol Costas arrays.

        m58 = numpy.concatenate( [ m79[7:36], m79[43:72] ] )

        # Demodulate the 58 8-FSK symbols into 174 bits.

        a174 = self.fsk_bits(m58)

        # The LDPC decoder wants log-likelihood estimates, indicating
        # how sure we are of each bit's value. This code isn't clever
        # enough to produce estimates, so it fakes them. 4.6 indicates
        # a zero, and -4.6 indicates a one.

        ## turn hard bits into 0.99 vs 0.01 log-likelihood,
        ## log_e( P(bit=0) / P(bit=1) )
        two = numpy.array([ 4.6, -4.6 ], dtype=numpy.int32)
        log174 = two[a174]

        # Call the LDPC decoder with the 174-bit codeword. The 
        # decoder has a big set of parity formulas that must be
        # satisfied by the bits in the codeword. Usually the codeword
        # contains errored bits, due to noise, interference, fading,
        # and badly aligned symbol sampling. The decoder tries to
        # guess which bits are incorrect, and flips them in an attempt
        # to cause the parity formulas to be satisfied.
        # If it succeeds, it returns 87 bits containing
        # the original message (the bits the sender handed its LDPC encoder).
        # Otherwise, after flipping
        # different combinations of bits for a while, it gives up.

        ## decode LDPC(174,87)
        a87 = ldpc_decode(log174)

        # A zero-length result array indicates that the decoder failed.

        if len(a87) == 0:
            ## failure.
            return None

        # The LDPC decode succeeded! FT8 double-checks the result with
        # a CRC. The CRC rarely fails if the LDPC
        # decode succeeded.

        ## check the CRC-12
        cksum = crc(numpy.append(a87[0:72], numpy.zeros(4, dtype=numpy.int32)),
                    crc12poly)
        if numpy.array_equal(cksum, a87[-12:]) == False:
            ## CRC failed.
            ## It's rare for LDPC to claim success but then CRC to fail.
            return None

        # The CRC succeeded, so it's highly likely that a87 contains
        # a correct message. Drop the 12 CRC bits and unpack the remainder into
        # a human-readable message, which process() will print.

        ## a87 is 75 bits of msg and 12 bits of CRC.
        a72 = a87[0:72]

        msg = self.unpack(a72)

        return msg

    # That's the end of the guts of the FT8 demodulator!

    # The remaining code is either not really part of
    # demodulation (e.g. the FT8 message format unpacker), or it's
    # fairly generic (the sound card and .wav file readers, and the
    # LDPC decoder).

    # Open the default sound card for input.

    def opencard(self):
        self.rate = 12000
        self.pya = pyaudio.PyAudio()
        self.card = self.pya.open(format=pyaudio.paInt16,
                                  ## input_device_index=XXX,
                                  channels=1,
                                  rate=self.rate,
                                  output=False,
                                  input=True)

    # gocard() reads samples from the sound card. Each time it
    # accumulates a full FT8 cycle's worth of samples,
    # starting at a cycle boundary, it passes them
    # to process(). gocard() calls process() in a separate thread,
    # because it needs to read samples for the next cycle while
    # process() is decoding.

    def gocard(self):
        buffered = numpy.array([], dtype=numpy.int16)
        while True:
            chunk = self.card.read(1024)
            chunk = numpy.frombuffer(chunk, dtype=numpy.int16)
            buffered = numpy.append(buffered, chunk)

            ## do we have all the samples for a full cycle?
            ## the nominal end of transmission occurs at 13.14 seconds.
            ## at that point we should have 

            sec = self.second()
            if sec >= 13.64 and len(buffered) > 13.64 * self.rate:
                ## it's the end of a cycle and we have enough samples.
                ## find the sample number at which the cycle started
                ## (i.e. 0.5 seconds before the nominal signal start time).
                start = len(buffered) - int(sec * self.rate)
                if start < 0:
                    start = 0
                samples = buffered[start:]

                ## decode in a separate thread, so that we can read
                ## sound samples for the next FT8 cycle while the
                ## thread decodes the cycle that just finished.
                th = threading.Thread(target=lambda s=samples: self.process(samples))
                th.daemon = True
                th.start()

                buffered = numpy.array([], dtype=numpy.int16)

    # second() returns the number of seconds since the start of the
    # current 15-second FT8 cycle.

    def second(self):
        t = time.time()
        dt = t - self.start_time
        dt /= 15.0
        return 15.0 * (dt - int(dt))

    # This program can read a .wav file instead of a sound card. The
    # .wav file must contain one FT8 cycle at 12000 samples/second.
    # That is the format that wsjt-x produces when it records audio.

    def openwav(self, filename):
        self.wav = wave.open(filename)
        self.rate = self.wav.getframerate()
        assert self.rate == 12000
        assert self.wav.getnchannels() == 1 ## one channel
        assert self.wav.getsampwidth() == 2 ## 16-bit audio

    def readwav(self, chan):
        frames = self.wav.readframes(8192)
        samples = numpy.frombuffer(frames, numpy.int16)
        return samples

    def gowav(self, filename, chan):
        self.openwav(filename)
        bufbuf = [ ]
        while True:
            buf = self.readwav(chan)
            if buf.size < 1:
                break
            bufbuf.append(buf)
        samples = numpy.concatenate(bufbuf)

        self.process(samples)

    # This code unpacks FT8 messages into human-readable
    # form. At a high level it interprets 72 bits of input as two call
    # signs and a grid or signal report.

    def unpack(self, a72):
        ## re-arrange the 72 bits into a format like JT65,
        ## for which this unpacker was originally written.
        a = [ ]
        for i in range(0, 72, 6):
            x = a72[i:i+6]
            y = (x[0] * 32 +
                 x[1] * 16 +
                 x[2] *  8 +
                 x[3] *  4 +
                 x[4] *  2 +
                 x[5] *  1)
            a.append(y)

        ## a[] has 12 0..63 symbols, or 72 bits.
        ## turn them into the original human-readable message.
        ## unpack([61, 37, 30, 28, 9, 27, 61, 58, 26, 3, 49, 16]) -> "G3LTF DL9KR JO40"
        nc1 = 0 ## 28 bits of first call
        nc1 |= a[4] >> 2 ## 4 bits
        nc1 |= a[3] << 4 ## 6 bits
        nc1 |= a[2] << 10 ## 6 bits
        nc1 |= a[1] << 16 ## 6 bits
        nc1 |= a[0] << 22 ## 6 bits

        nc2 = 0 ## 28 bits of second call
        nc2 |= (a[4] & 3) << 26 ## 2 bits
        nc2 |= a[5] << 20 ## 6 bits
        nc2 |= a[6] << 14 ## 6 bits
        nc2 |= a[7] << 8 ## 6 bits
        nc2 |= a[8] << 2 ## 6 bits
        nc2 |= a[9] >> 4 ## 2 bits

        ng = 0 ## 16 bits of grid
        ng |= (a[9] & 15) << 12 ## 4 bits
        ng |= a[10] << 6 ## 6 bits
        ng |= a[11]

        if ng >= 32768:
            txt = self.unpacktext(nc1, nc2, ng)
            return txt

        NBASE = 37*36*10*27*27*27

        if nc1 == NBASE+1:
            c2 = self.unpackcall(nc2)
            grid = self.unpackgrid(ng)
            return "CQ %s %s" % (c2, grid)

        if nc1 >= 267649090 and nc1 <= 267698374:
            ## CQ with suffix (e.g. /QRP)
            n = nc1 - 267649090
            sf = self.charn(n % 37)
            n /= 37
            sf = self.charn(n % 37) + sf
            n /= 37
            sf = self.charn(n % 37) + sf
            n /= 37
            c2 = self.unpackcall(nc2)
            grid = self.unpackgrid(ng)
            return "CQ %s/%s %s" % (c2, sf, grid)

        c1 = self.unpackcall(nc1)
        if c1 == "CQ9DX ":
            c1 = "CQ DX "
        m = re.match(r'^ *E9([A-Z][A-Z]) *$', c1)
        if m != None:
            c1 = "CQ " + m.group(1)
        c2 = self.unpackcall(nc2)
        grid = self.unpackgrid(ng)
        msg = "%s %s %s" % (c1, c2, grid)

        if "000AAA" in msg:
            return None

        return msg

    ## convert packed character to Python string.
    ## 0..9 a..z space
    def charn(self, c):
        if c >= 0 and c <= 9:
            return chr(ord('0') + c)
        if c >= 10 and c < 36:
            return chr(ord('A') + c - 10)
        if c == 36:
            return ' '
        ## sys.stderr.write("jt65 charn(%d) bad\n" % (c))
        return '?'

    ## x is an integer, e.g. nc1 or nc2, containing all the
    ## call sign bits from a packed message.
    ## 28 bits.
    def unpackcall(self, x):
        a = [ 0, 0, 0, 0, 0, 0 ]
        a[5] = self.charn((x % 27) + 10) ## + 10 b/c only alpha+space
        x = int(x / 27)
        a[4] = self.charn((x % 27) + 10)
        x = int(x / 27)
        a[3] = self.charn((x % 27) + 10)
        x = int(x / 27)
        a[2] = self.charn(x%10) ## digit only
        x = int(x / 10)
        a[1] = self.charn(x % 36) ## letter or digit
        x = int(x / 36)
        a[0] = self.charn(x)
        return ''.join(a)

    ## extract maidenhead locator
    def unpackgrid(self, ng):
        ## start of special grid locators for sig strength &c.
        NGBASE = 180*180

        if ng == NGBASE+1:
            return "    "
        if ng >= NGBASE+1 and ng < NGBASE+31:
            return " -%02d" % (ng - (NGBASE+1)) ## sig str, -01 to -30 DB
        if ng >= NGBASE+31 and ng < NGBASE+62:
            return "R-%02d" % (ng - (NGBASE+31))
        if ng == NGBASE+62:
            return "RO  "
        if ng == NGBASE+63:
            return "RRR "
        if ng == NGBASE+64:
            return "73  "

        lat = (ng % 180) - 90
        ng = int(ng / 180)
        lng = (ng * 2) - 180

        g = "%c%c%c%c" % (ord('A') + int((179-lng)/20),
                          ord('A') + int((lat+90)/10),
                          ord('0') + int(((179-lng)%20)/2),
                          ord('0') + (lat+90)%10)

        if g[0:2] == "KA":
            ## really + signal strength
            sig = int(g[2:4]) - 50
            return "+%02d" % (sig)

        if g[0:2] == "LA":
            ## really R+ signal strength
            sig = int(g[2:4]) - 50
            return "R+%02d" % (sig)

        return g

    def unpacktext(self, nc1, nc2, nc3):
        c = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ +-./?"

        nc3 &= 32767
        if (nc1 & 1) != 0:
            nc3 += 32768
        nc1 >>= 1
        if (nc2 & 1) != 0:
            nc3 += 65536
        nc2 >>= 1

        msg = [""] * 22

        for i in range(4, -1, -1):
            j = nc1 % 42
            msg[i] = c[j]
            nc1 = nc1 // 42

        for i in range(9, 4, -1):
            j = nc2 % 42
            msg[i] = c[j]
            nc2 = nc2 // 42

        for i in range(12, 9, -1):
            j = nc3 % 42
            msg[i] = c[j]
            nc3 = nc3 // 42

        return ''.join(msg)

    def __init__(self):
        self.block = 1920 ## samples per FT8 symbol, at 12000 samples/second

        ## set self.start_time to the UNIX time of the start
        ## of the last UTC minute.
        now = int(time.time())
        gm = time.gmtime(now)
        self.start_time = now - gm.tm_sec

# Now comes the LDPC decoder. The decoder is driven by tables that
# describe the parity checks that the codeword must satify.

# Each row of Nm describes one parity check.
# Each number is an index into the codeword (1-origin).
# The codeword bits mentioned in each row must exclusive-or to zero.
# There are 87 rows.
# Nm is a copy of wsjt-x's bpdecode174.f90.
Nm = [
    [ 1,   30,  60,  89,   118,  147,  0 ],
    [ 2,   31,  61,  90,   119,  147,  0 ],
    [ 3,   32,  62,  91,   120,  148,  0 ],
    [ 4,   33,  63,  92,   121,  149,  0 ],
    [ 2,   34,  64,  93,   122,  150,  0 ],
    [ 5,   33,  65,  94,   123,  148,  0 ],
    [ 6,   34,  66,  95,   124,  151,  0 ],
    [ 7,   35,  67,  96,   120,  152,  0 ],
    [ 8,   36,  68,  97,   125,  153,  0 ],
    [ 9,   37,  69,  98,   126,  152,  0 ],
    [ 10,  38,  70,  99,   127,  154,  0 ],
    [ 11,  39,  71,  100,  126,  155,  0 ],
    [ 12,  40,  61,  101,  128,  145,  0 ],
    [ 10,  33,  60,  95,   128,  156,  0 ],
    [ 13,  41,  72,  97,   126,  157,  0 ],
    [ 13,  42,  73,  90,   129,  156,  0 ],
    [ 14,  39,  74,  99,   130,  158,  0 ],
    [ 15,  43,  75,  102,  131,  159,  0 ],
    [ 16,  43,  71,  103,  118,  160,  0 ],
    [ 17,  44,  76,  98,   130,  156,  0 ],
    [ 18,  45,  60,  96,   132,  161,  0 ],
    [ 19,  46,  73,  83,   133,  162,  0 ],
    [ 12,  38,  77,  102,  134,  163,  0 ],
    [ 19,  47,  78,  104,  135,  147,  0 ],
    [ 1,   32,  77,  105,  136,  164,  0 ],
    [ 20,  48,  73,  106,  123,  163,  0 ],
    [ 21,  41,  79,  107,  137,  165,  0 ],
    [ 22,  42,  66,  108,  138,  152,  0 ],
    [ 18,  42,  80,  109,  139,  154,  0 ],
    [ 23,  49,  81,  110,  135,  166,  0 ],
    [ 16,  50,  82,  91,   129,  158,  0 ],
    [ 3,   48,  63,  107,  124,  167,  0 ],
    [ 6,   51,  67,  111,  134,  155,  0 ],
    [ 24,  35,  77,  100,  122,  162,  0 ],
    [ 20,  45,  76,  112,  140,  157,  0 ],
    [ 21,  36,  64,  92,   130,  159,  0 ],
    [ 8,   52,  83,  111,  118,  166,  0 ],
    [ 21,  53,  84,  113,  138,  168,  0 ],
    [ 25,  51,  79,  89,   122,  158,  0 ],
    [ 22,  44,  75,  107,  133,  155,  172 ],
    [ 9,   54,  84,  90,   141,  169,  0 ],
    [ 22,  54,  85,  110,  136,  161,  0 ],
    [ 8,   37,  65,  102,  129,  170,  0 ],
    [ 19,  39,  85,  114,  139,  150,  0 ],
    [ 26,  55,  71,  93,   142,  167,  0 ],
    [ 27,  56,  65,  96,   133,  160,  174 ],
    [ 28,  31,  86,  100,  117,  171,  0 ],
    [ 28,  52,  70,  104,  132,  144,  0 ],
    [ 24,  57,  68,  95,   137,  142,  0 ],
    [ 7,   30,  72,  110,  143,  151,  0 ],
    [ 4,   51,  76,  115,  127,  168,  0 ],
    [ 16,  45,  87,  114,  125,  172,  0 ],
    [ 15,  30,  86,  115,  123,  150,  0 ],
    [ 23,  46,  64,  91,   144,  173,  0 ],
    [ 23,  35,  75,  113,  145,  153,  0 ],
    [ 14,  41,  87,  108,  117,  149,  170 ],
    [ 25,  40,  85,  94,   124,  159,  0 ],
    [ 25,  58,  69,  116,  143,  174,  0 ],
    [ 29,  43,  61,  116,  132,  162,  0 ],
    [ 15,  58,  88,  112,  121,  164,  0 ],
    [ 4,   59,  72,  114,  119,  163,  173 ],
    [ 27,  47,  86,  98,   134,  153,  0 ],
    [ 5,   44,  78,  109,  141,  0,    0 ],
    [ 10,  46,  69,  103,  136,  165,  0 ],
    [ 9,   50,  59,  93,   128,  164,  0 ],
    [ 14,  57,  58,  109,  120,  166,  0 ],
    [ 17,  55,  62,  116,  125,  154,  0 ],
    [ 3,   54,  70,  101,  140,  170,  0 ],
    [ 1,   36,  82,  108,  127,  174,  0 ],
    [ 5,   53,  81,  105,  140,  0,    0 ],
    [ 29,  53,  67,  99,   142,  173,  0 ],
    [ 18,  49,  74,  97,   115,  167,  0 ],
    [ 2,   57,  63,  103,  138,  157,  0 ],
    [ 26,  38,  79,  112,  135,  171,  0 ],
    [ 11,  52,  66,  88,   119,  148,  0 ],
    [ 20,  40,  68,  117,  141,  160,  0 ],
    [ 11,  48,  81,  89,   146,  169,  0 ],
    [ 29,  47,  80,  92,   146,  172,  0 ],
    [ 6,   32,  87,  104,  145,  169,  0 ],
    [ 27,  34,  74,  106,  131,  165,  0 ],
    [ 12,  56,  84,  88,   139,  0,    0 ],
    [ 13,  56,  62,  111,  146,  171,  0 ],
    [ 26,  37,  80,  105,  144,  151,  0 ],
    [ 17,  31,  82,  113,  121,  161,  0 ],
    [ 28,  49,  59,  94,   137,  0,    0 ],
    [ 7,   55,  83,  101,  131,  168,  0 ],
    [ 24,  50,  78,  106,  143,  149,  0 ],
]

# Mn is the dual of Nm.
# Each row corresponds to a codeword bit.
# The numbers indicate which three parity
# checks (rows in Nm) refer to the codeword bit.
# 1-origin.
# Mn is a copy of wsjt-x's bpdecode174.f90.
Mn = [
  [ 1, 25, 69 ],
  [ 2, 5, 73 ],
  [ 3, 32, 68 ],
  [ 4, 51, 61 ],
  [ 6, 63, 70 ],
  [ 7, 33, 79 ],
  [ 8, 50, 86 ],
  [ 9, 37, 43 ],
  [ 10, 41, 65 ],
  [ 11, 14, 64 ],
  [ 12, 75, 77 ],
  [ 13, 23, 81 ],
  [ 15, 16, 82 ],
  [ 17, 56, 66 ],
  [ 18, 53, 60 ],
  [ 19, 31, 52 ],
  [ 20, 67, 84 ],
  [ 21, 29, 72 ],
  [ 22, 24, 44 ],
  [ 26, 35, 76 ],
  [ 27, 36, 38 ],
  [ 28, 40, 42 ],
  [ 30, 54, 55 ],
  [ 34, 49, 87 ],
  [ 39, 57, 58 ],
  [ 45, 74, 83 ],
  [ 46, 62, 80 ],
  [ 47, 48, 85 ],
  [ 59, 71, 78 ],
  [ 1, 50, 53 ],
  [ 2, 47, 84 ],
  [ 3, 25, 79 ],
  [ 4, 6, 14 ],
  [ 5, 7, 80 ],
  [ 8, 34, 55 ],
  [ 9, 36, 69 ],
  [ 10, 43, 83 ],
  [ 11, 23, 74 ],
  [ 12, 17, 44 ],
  [ 13, 57, 76 ],
  [ 15, 27, 56 ],
  [ 16, 28, 29 ],
  [ 18, 19, 59 ],
  [ 20, 40, 63 ],
  [ 21, 35, 52 ],
  [ 22, 54, 64 ],
  [ 24, 62, 78 ],
  [ 26, 32, 77 ],
  [ 30, 72, 85 ],
  [ 31, 65, 87 ],
  [ 33, 39, 51 ],
  [ 37, 48, 75 ],
  [ 38, 70, 71 ],
  [ 41, 42, 68 ],
  [ 45, 67, 86 ],
  [ 46, 81, 82 ],
  [ 49, 66, 73 ],
  [ 58, 60, 66 ],
  [ 61, 65, 85 ],
  [ 1, 14, 21 ],
  [ 2, 13, 59 ],
  [ 3, 67, 82 ],
  [ 4, 32, 73 ],
  [ 5, 36, 54 ],
  [ 6, 43, 46 ],
  [ 7, 28, 75 ],
  [ 8, 33, 71 ],
  [ 9, 49, 76 ],
  [ 10, 58, 64 ],
  [ 11, 48, 68 ],
  [ 12, 19, 45 ],
  [ 15, 50, 61 ],
  [ 16, 22, 26 ],
  [ 17, 72, 80 ],
  [ 18, 40, 55 ],
  [ 20, 35, 51 ],
  [ 23, 25, 34 ],
  [ 24, 63, 87 ],
  [ 27, 39, 74 ],
  [ 29, 78, 83 ],
  [ 30, 70, 77 ],
  [ 31, 69, 84 ],
  [ 22, 37, 86 ],
  [ 38, 41, 81 ],
  [ 42, 44, 57 ],
  [ 47, 53, 62 ],
  [ 52, 56, 79 ],
  [ 60, 75, 81 ],
  [ 1, 39, 77 ],
  [ 2, 16, 41 ],
  [ 3, 31, 54 ],
  [ 4, 36, 78 ],
  [ 5, 45, 65 ],
  [ 6, 57, 85 ],
  [ 7, 14, 49 ],
  [ 8, 21, 46 ],
  [ 9, 15, 72 ],
  [ 10, 20, 62 ],
  [ 11, 17, 71 ],
  [ 12, 34, 47 ],
  [ 13, 68, 86 ],
  [ 18, 23, 43 ],
  [ 19, 64, 73 ],
  [ 24, 48, 79 ],
  [ 25, 70, 83 ],
  [ 26, 80, 87 ],
  [ 27, 32, 40 ],
  [ 28, 56, 69 ],
  [ 29, 63, 66 ],
  [ 30, 42, 50 ],
  [ 33, 37, 82 ],
  [ 35, 60, 74 ],
  [ 38, 55, 84 ],
  [ 44, 52, 61 ],
  [ 51, 53, 72 ],
  [ 58, 59, 67 ],
  [ 47, 56, 76 ],
  [ 1, 19, 37 ],
  [ 2, 61, 75 ],
  [ 3, 8, 66 ],
  [ 4, 60, 84 ],
  [ 5, 34, 39 ],
  [ 6, 26, 53 ],
  [ 7, 32, 57 ],
  [ 9, 52, 67 ],
  [ 10, 12, 15 ],
  [ 11, 51, 69 ],
  [ 13, 14, 65 ],
  [ 16, 31, 43 ],
  [ 17, 20, 36 ],
  [ 18, 80, 86 ],
  [ 21, 48, 59 ],
  [ 22, 40, 46 ],
  [ 23, 33, 62 ],
  [ 24, 30, 74 ],
  [ 25, 42, 64 ],
  [ 27, 49, 85 ],
  [ 28, 38, 73 ],
  [ 29, 44, 81 ],
  [ 35, 68, 70 ],
  [ 41, 63, 76 ],
  [ 45, 49, 71 ],
  [ 50, 58, 87 ],
  [ 48, 54, 83 ],
  [ 13, 55, 79 ],
  [ 77, 78, 82 ],
  [ 1, 2, 24 ],
  [ 3, 6, 75 ],
  [ 4, 56, 87 ],
  [ 5, 44, 53 ],
  [ 7, 50, 83 ],
  [ 8, 10, 28 ],
  [ 9, 55, 62 ],
  [ 11, 29, 67 ],
  [ 12, 33, 40 ],
  [ 14, 16, 20 ],
  [ 15, 35, 73 ],
  [ 17, 31, 39 ],
  [ 18, 36, 57 ],
  [ 19, 46, 76 ],
  [ 21, 42, 84 ],
  [ 22, 34, 59 ],
  [ 23, 26, 61 ],
  [ 25, 60, 65 ],
  [ 27, 64, 80 ],
  [ 30, 37, 66 ],
  [ 32, 45, 72 ],
  [ 38, 51, 86 ],
  [ 41, 77, 79 ],
  [ 43, 56, 68 ],
  [ 47, 74, 82 ],
  [ 40, 52, 78 ],
  [ 54, 61, 71 ],
  [ 46, 58, 69 ],
]

# This is an indirection table that moves a
# codeword's 87 systematic (message) bits to the end.
# It's copied from the wsjt-x source.
colorder = [
  0, 1, 2, 3, 30, 4, 5, 6, 7, 8, 9, 10, 11, 32, 12, 40, 13, 14, 15, 16,
  17, 18, 37, 45, 29, 19, 20, 21, 41, 22, 42, 31, 33, 34, 44, 35, 47,
  51, 50, 43, 36, 52, 63, 46, 25, 55, 27, 24, 23, 53, 39, 49, 59, 38,
  48, 61, 60, 57, 28, 62, 56, 58, 65, 66, 26, 70, 64, 69, 68, 67, 74,
  71, 54, 76, 72, 75, 78, 77, 80, 79, 73, 83, 84, 81, 82, 85, 86, 87,
  88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
  104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
  118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
  132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
  146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173
  ]

# The LDPC decoder function.
# Given a 174-bit codeword as an array of log-likelihood of zero,
# return an 87-bit plain text, or zero-length array.
# The algorithm is the sum-product algorithm
# from Sarah Johnson's Iterative Error Correction book.
## codeword[i] = log ( P(x=0) / P(x=1) )
def ldpc_decode(codeword):
    ## 174 codeword bits
    ## 87 parity checks

    mnx = numpy.array(Mn, dtype=numpy.int32)
    nmx = numpy.array(Nm, dtype=numpy.int32)

    ## Mji
    ## each codeword bit i tells each parity check j
    ## what the bit's log-likelihood of being 0 is
    ## based on information *other* than from that
    ## parity check.
    m = numpy.zeros((87, 174))

    for i in range(0, 174):
        for j in range(0, 87):
            m[j][i] = codeword[i]

    for iter in range(0, 30):
        ## Eji
        ## each check j tells each codeword bit i the
        ## log likelihood of the bit being zero based
        ## on the *other* bits in that check.
        e = numpy.zeros((87, 174))

        ## messages from checks to bits.
        ## for each parity check
        ##for j in range(0, 87):
        ##    # for each bit mentioned in this parity check
        ##    for i in Nm[j]:
        ##        if i <= 0:
        ##            continue
        ##        a = 1
        ##        # for each other bit mentioned in this parity check
        ##        for ii in Nm[j]:
        ##            if ii != i:
        ##                a *= math.tanh(m[j][ii-1] / 2.0)
        ##        e[j][i-1] = math.log((1 + a) / (1 - a))
        for i in range(0, 7):
            a = numpy.ones(87)
            for ii in range(0, 7):
                if ii != i:
                    x1 = numpy.tanh(m[range(0, 87), nmx[:,ii]-1] / 2.0)
                    x2 = numpy.where(numpy.greater(nmx[:,ii], 0.0), x1, 1.0)
                    a = a * x2
            ## avoid divide by zero, i.e. a[i]==1.0
            ## XXX why is a[i] sometimes 1.0?
            b = numpy.where(numpy.less(a, 0.99999), a, 0.99)
            c = numpy.log((b + 1.0) / (1.0 - b))
            ## have assign be no-op when nmx[a,b] == 0
            d = numpy.where(numpy.equal(nmx[:,i], 0),
                            e[range(0,87), nmx[:,i]-1],
                            c)
            e[range(0,87), nmx[:,i]-1] = d

        ## decide if we are done -- compute the corrected codeword,
        ## see if the parity check succeeds.
        ## sum the three log likelihoods contributing to each codeword bit.
        e0 = e[mnx[:,0]-1, range(0,174)]
        e1 = e[mnx[:,1]-1, range(0,174)]
        e2 = e[mnx[:,2]-1, range(0,174)]
        ll = codeword + e0 + e1 + e2
        ## log likelihood > 0 => bit=0.
        cw = numpy.select( [ ll < 0 ], [ numpy.ones(174, dtype=numpy.int32) ])
        if ldpc_check(cw):
            ## success!
            ## it's a systematic code, though the plain-text bits are scattered.
            ## collect them.
            decoded = cw[colorder]
            decoded = decoded[-87:]
            return decoded

        ## messages from bits to checks.
        for j in range(0, 3):
            ## for each column in Mn.
            ll = codeword
            if j != 0:
                e0 = e[mnx[:,0]-1, range(0,174)]
                ll = ll + e0
            if j != 1:
                e1 = e[mnx[:,1]-1, range(0,174)]
                ll = ll + e1
            if j != 2:
                e2 = e[mnx[:,2]-1, range(0,174)]
                ll = ll + e2
            m[mnx[:,j]-1, range(0,174)] = ll

    ## could not decode.
    return numpy.array([])

# A helper function to decide if
# a 174-bit codeword passes the LDPC parity checks.
def ldpc_check(codeword):
    for e in Nm:
        x = 0
        for i in e:
            if i != 0:
                x ^= codeword[i-1]
        if x != 0:
            return False
    return True

# The CRC-12 polynomial, copied from wsjt-x's 0xc06.
crc12poly = [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 ]

# crc() is a copy of Evan Sneath's code, from
# https://gist.github.com/evansneath/4650991 .
# div is crc12poly.

##
##
## generate with x^3 + x + 1:
##   >>> xc.crc([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1])
##   array([1, 0, 0])
## check:
##   >>> xc.crc([1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0])
##   array([0, 0, 0])
##
## 0xc06 is really 0x1c06 or [ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0 ]
##
def crc(msg, div):
    ## Append the code to the message. If no code is given, default to '000'
    code = numpy.zeros(len(div)-1, dtype=numpy.int32)
    assert len(code) == len(div) - 1
    msg = numpy.append(msg, code)

    div = numpy.array(div, dtype=numpy.int32)
    divlen = len(div)

    ## Loop over every message bit (minus the appended code)
    for i in range(len(msg)-len(code)):
        ## If that messsage bit is 1, perform modulo 2 multiplication
        if msg[i] == 1:
            ##for j in range(len(div)):
            ##    # Perform modulo 2 multiplication on each index of the divisor
            ##    msg[i+j] = (msg[i+j] + div[j]) % 2
            msg[i:i+divlen] = numpy.mod(msg[i:i+divlen] + div, 2)

    ## Output the last error-checking code portion of the message generated
    return msg[-len(code):]

# The main function: look at the command-line arguments, decide
# whether to read from a sound card or from .wav files.

def main():
    if len(sys.argv) == 1:
        r = FT8()
        r.opencard()
        r.gocard()
        sys.exit(0)

    i = 1
    while i < len(sys.argv):
        r = FT8()
        r.gowav(sys.argv[i], 0)
        i += 1

if __name__ == '__main__':
    main()
