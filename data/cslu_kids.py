import collections
import os
import re
import wave
from enum import Enum
from typing import List

Utterance = collections.namedtuple('Utterance', [
    'utterance_id',
    'speaker_id',
    'filename',
    'transcript'
])


ID_PATTERN = "(ks([0-9a-l])([0-9a-z]{2})([0-9a-z]{2})0)"
WAV_PATTERN = re.compile(ID_PATTERN + ".wav")
TRANSCRIPT_PATTERN = re.compile(ID_PATTERN + ".txt")
VERIFIED_PATTERN = re.compile("^.*/{}.wav ([1-4])$".format(ID_PATTERN))
ALL_MAP_PATTERN = re.compile("([A-Z0-9]{2}) \"(.*)\"")


class Quality(Enum):
    '''
    (From corpus documentation)

    Description of verification:

    1 Good: Only the target word is said.
    2 Maybe: Target word is present, but there's other junk in the file.
    3 Bad: Target word is not said.
    4 Puff: Same as good, but w/ an air puff.

    Verification is for scripted speech ONLY!
    '''
    GOOD = "1"
    MAYBE = "2"
    BAD = "3"
    PUFF = "4"


class Grade(Enum):
    G00 = 0
    G01 = 1
    G02 = 2
    G03 = 3
    G04 = 4
    G05 = 5
    G06 = 6
    G07 = 7
    G08 = 8
    G09 = 9
    G10 = 10


def get_all(
        ogi_path,
        *_,
        vocabulary=None,
        oov_word="<UNK>",
        quality=(Quality.GOOD,),
        grades=(Grade.G00, Grade.G01, Grade.G02, Grade.G03, Grade.G04, Grade.G05, Grade.G06, Grade.G07, Grade.G08, Grade.G09, Grade.G10),
        spontaneous=True,
        scripted=True,
        max_length=None,
        prefix='cslu_',
        lowercase=True
) -> List[Utterance]:
    if grades is None:
        grades = (Grade.G00, Grade.G01, Grade.G02, Grade.G03, Grade.G04, Grade.G05, Grade.G06, Grade.G07, Grade.G08, Grade.G09, Grade.G10)
    wavs_path = os.path.join(ogi_path, "speech")
    transcripts_path = os.path.join(ogi_path, "trans")
    docs_path = os.path.join(ogi_path, "docs")
    audios = _get_audios(wavs_path, grades)
    transcripts = {}
    if spontaneous:
        spontaneous_transcripts = _get_spontaneous_transcripts(transcripts_path, vocabulary, oov_word)
        transcripts.update(spontaneous_transcripts)
    if scripted:
        scripted_transcripts = _get_scripted_transcripts(docs_path, vocabulary, oov_word, quality)
        transcripts.update(scripted_transcripts)
    if lowercase:
        transcripts = { k: v.lower() for k, v in transcripts.items() }
    audios = _remove_some(audios, transcripts=transcripts, max_length=max_length)

    get_utterance = lambda id:  Utterance(
        utterance_id=prefix + id,
        speaker_id=prefix + id[:5],
        filename=audios[id],
        transcript = transcripts[id],
    )

    return [get_utterance(id) for id in transcripts
            if id in audios and transcripts[id] is not None]


def _get_grade(gender_grade_code):
    ascii_offset = ord(gender_grade_code) - ord("b")
    if ascii_offset >= 0:
        return Grade(ascii_offset)
    elif gender_grade_code == "a":
        return Grade.G10  # "a" means 10th grade
    else:
        return Grade(int(gender_grade_code))  # gender_grade_code is numerical value


def _get_audios(path, grades):
    audios = {}
    for file, (utterance_id, gender_grade_code, speaker_id, prompt_id) in search_dir(path, WAV_PATTERN):
        if _get_grade(gender_grade_code) not in grades:
            continue
        audios[utterance_id] = file
    return audios


def _get_spontaneous_transcripts(path, vocabulary, oov_word):
    transcripts = {}
    oov = set()
    for file, (utterance_id, gender_grade_code, speaker_id, prompt_id) in search_dir(path, TRANSCRIPT_PATTERN):
        with open(file, "r") as file:
            transcript = []
            for line in file:
                sanitized, line_oov = _sanitize(line, vocabulary, oov_word)
                oov.update(line_oov)
                transcript.extend(sanitized)
            transcripts[utterance_id] = " ".join(transcript)
    for word in oov:
        print("Labeled as {}: {}".format(oov_word, word))
    return transcripts


def _get_prompts(docs_path, vocabulary, oov_word):
    prompts = {}
    prompts["8v"] = "abnormal" # missing from all.map
    oov = set()
    with open(os.path.join(docs_path, "all.map"), "r") as file:
        for line in file:
            for prompt_id, transcript in re.findall(ALL_MAP_PATTERN, line):
                sanitized, line_oov = _sanitize(transcript, vocabulary, oov_word)
                oov.update(line_oov)
                prompts[prompt_id.lower()] = " ".join(sanitized)
    for word in oov:
        print("Labeled as {}: {}".format(oov_word, word))
    return prompts


def _get_scripted_transcripts(docs_path, vocabulary, oov_word, quality):
    prompts = _get_prompts(docs_path, vocabulary, oov_word)
    transcripts = {}
    files = [f for f in os.listdir(docs_path) if f.endswith("verified.txt")]
    for file in files:
        with open(os.path.join(docs_path, file), "r") as file:
            for line in file:
                for utterance_id, _, _, prompt_id, q in re.findall(VERIFIED_PATTERN, line):
                    if Quality(q) not in quality:
                        continue
                    if prompt_id not in prompts:
                        print("Prompt {} not found in docs/all.map.".format(prompt_id))
                        continue
                    sanitized, line_oov = _sanitize(prompts[prompt_id], vocabulary, oov_word)
                    transcripts[utterance_id] = ' '.join(sanitized)
    return transcripts


def _remove_some(audios, transcripts, max_length):
    to_remove = set()
    for key, full_path in audios.items():
        if key not in transcripts:
            to_remove.add(key)
            continue
        if max_length:
            with wave.open(full_path, 'r') as wav_file:
                file_length = wav_file.getnframes() / wav_file.getframerate()
                if file_length > max_length:
                    print("Skipping {}, too long ({:.2f} sec > {} sec)".format(
                        full_path,
                        file_length,
                        max_length
                    ))
                    to_remove.add(key)
                    continue
    return { k: v for k, v in audios.items() if k not in to_remove }


def _sanitize(line, vocabulary, oov_word):
    out = []
    oov = []
    words = re.sub("(?<! )<", " <", line).split() # Tokenize word<noise> as two separate tokens

    for word in words:
        word = word.strip(",")
        word = word.strip("'")
        word = word.strip("*")
        word = word.upper().strip("!")

        if vocabulary is not None and (word.upper() not in vocabulary and word.lower() not in vocabulary):
            out.append(oov_word)
            oov.append(word)
            continue

        out.append(word)

    return out, oov


def search_dir(path, pattern, match_full_path=False):
    for basedir, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(basedir, file)
            if match_full_path:
                file = file_path
            for match in re.findall(pattern, file):
                yield file_path, match
