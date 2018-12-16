#!/usr/bin/python

#
# spectral_shuffle.py
#
# License: Public Domain (www.unlicense.org)
#
# This is free and unencumbered software released into the public domain.
# Anyone is free to copy, modify, publish, use, compile, sell, or distribute this 
# software, either in source code form or as a compiled binary, for any purpose, 
# commercial or non-commercial, and by any means.
# In jurisdictions that recognize copyright laws, the author or authors of this 
# software dedicate any and all copyright interest in the software to the public 
# domain. We make this dedication for the benefit of the public at large and to 
# the detriment of our heirs and successors. We intend this dedication to be an 
# overt act of relinquishment in perpetuity of all present and future rights to 
# this software under copyright law.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN 
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import scipy.io.wavfile as wave
import numpy as np
import sys

BUFFER_SIZE = 2048 * 2

argv = sys.argv[1:]
if len(argv) < 2:
    print('usage: spectral_shuffle.py in out')
    sys.exit(1)

rate, src = wave.read(argv[0])

cutoff = 2000
copied = np.zeros(src.size, dtype=src.dtype)
i, j = 0, BUFFER_SIZE
step = BUFFER_SIZE / 2
maps = []
lens = []
while i < src.size:
    s = i
    e = min(src.size, j)
    n = e - s
    spectrum = np.fft.rfft(src[s:e] * np.hanning(n), BUFFER_SIZE)
    freqs = np.fft.rfftfreq(BUFFER_SIZE, 1.0/rate)
    # spectrum[freqs > cutoff] = 0 # zeroing bin
    maps.append(spectrum)
    lens.append((s, e))
    # samples = np.fft.irfft(spectrum)
    # copied[s:e] += samples[0:n]
    i += step
    j += step

def smoothstep(a, b, t):
    def lerp(a, b, t):
        return a*(1.0-t) + b*t
    return lerp(a, b, t * t * (3.0 - 2.0*t))

for i in range(1, len(maps)):
    x = np.random.randint(0, len(maps))
    for k in range(20):
        t = k / 20.0
        spectrum = smoothstep(maps[x], maps[(x+1)%len(maps)], np.random.uniform(0.0, 1.0))
        s = int(smoothstep(lens[i-1][0], lens[i][0], t))
        e = int(smoothstep(lens[i-1][1], lens[i][1], t))
        n = e - s
        samples = np.fft.irfft(spectrum)
        copied[s:e] += samples[0:n] * (1.0/4.0)

wave.write(argv[1], rate, copied)
