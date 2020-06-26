import numpy as np
import sys,os,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypianoroll import Multitrack, Track

def write_midi(filepath, pianorolls, program_nums=None, is_drums=None,
               track_names=None, velocity=100, tempo=40.0, beat_resolution=24):

    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]
    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same"
                         "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same"
                         "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        #plt.subplot(10,1,idx+1)
        #plt.imshow(pianorolls[..., idx].T,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    #plt.savefig(cf.MP3Name)
    multitrack.write(filepath)

def main(arg):
    name = arg[1]
    #load data
    total = np.transpose(np.load('output_data/roll/'+name),(2,1,0))

    plt.subplot(9,1,1)
    plt.imshow(total[:,:,0],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,2)
    plt.imshow(total[:,:,1],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,3)
    plt.imshow(total[:,:,2],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,4)
    plt.imshow(total[:,:,3],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,5)
    plt.imshow(total[:,:,4],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,6)
    plt.imshow(total[:,:,5],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,7)
    plt.imshow(total[:,:,6],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,8)
    plt.imshow(total[:,:,7],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.subplot(9,1,9)
    plt.imshow(total[:,:,8],cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.savefig('tmp.png')

    s = total.shape
    pad1 = np.zeros((s[0],21,s[2]))
    pad2 = np.zeros((s[0],19,s[2]))
    total = np.concatenate((pad1, total), axis=1)
    total = np.concatenate((total, pad2), axis=1).astype(bool)
    #decide midi code
    midi_code = [2,25,28,56,66,34,41,43,74]
    isDrum = [False,False,False,False,False,False,False,False,False]
    track_name = ['Piano','A-Guitar','E-Guitar','Trumpet','Saxphone','Bass','Violin','Cello','Flute']
    write_midi('output_data/midi/'+name[:-4],total,midi_code,isDrum,track_name)

if __name__ == "__main__":
    main(sys.argv)