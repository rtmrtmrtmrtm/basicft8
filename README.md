# basicft8

basicft8.py is a simple demodulator for the WSJT-X FT8 mode. It misses many
possible decodes, but in return it's relatively easy to understand.
You can see a detailed explanation of the code at
http://www.rtmrtm.org/basicft8/ .
Please feel free to copy and modify this code for your own amusement.
I've found that writing my own demodulators for FT8, JT65, etc
has been very interesting and educational.

The code is written in Python 2.7. You'll need a few extra Python
packages. Here's how to install them on Ubuntu Linux:
```
  sudo apt-get install python2.7
  sudo apt-get install python-numpy
  sudo apt-get install python-pyaudio
```

If you have a Mac with macports:
```
  sudo port install python27
  sudo port install py27-numpy
  sudo port install py27-pyaudio
```

This repository contains a few sample FT8 .wav files to help test the
demodulator. To read one, provide the file name on the command line,
like this:

```
  python basicft8.py samples/170923_082000.wav
```

(Perhaps python2 or python27 or python2.7 instead of python.)
When I do this, I see two lines of output corresponding to two decodes:
```
  1962.5 CQ FK8HA  RG37
  1237.5 ZL1RPL YV5DRN RRR 
```

There are at least seven more signals that could be decoded, but that
would require more sophisticated demodulation algorithms.

To listen to audio from a sound card (presumably hooked up to a radio
set to e.g. 14.074 USB), try this:

```
  python basicft8.py
```

By default basicft8.py tries to read audio from sound card number 2.
To use a different sound card, find the definition of cardno in the
source and change it. The sound card has to support an input rate of
12,000 samples/second.
