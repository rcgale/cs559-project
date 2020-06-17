def read_sound_file(filename):
    try:
        import soundfile
        audio, fs = soundfile.read(filename)  # <-- flac support
    except:
        import scipy.io.wavfile
        fs, audio = scipy.io.wavfile.read(filename)
    return fs, audio
