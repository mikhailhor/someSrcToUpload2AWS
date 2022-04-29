import csv
import os
import argparse
import librosa
import soundfile as sf
import tqdm

# Set Inputs
csvpath = r"/media/ali/G2/Data/pl_PL/by_book/male/piotr_nater/lalka/metadata.csv"
outputpath = "PolandTTS/Prepared"
outputfilename = "metadata.txt"
audiopath = "/media/ali/PrjPrgm/Projects/CoguiTTS/PolandTTS/Prepared/wavs"

# format the data and save it in the meta_data.txt for using Congui-TTS
def format_data():

     with open(os.path.join(outputpath, outputfilename), 'w') as fwrite:

        with open (csvpath) as fd:
            rd = csv.reader ( fd, delimiter="\t", quotechar='"' )
            idx = 0
            for row in rd:
                try:
                    audio_name = row[ 0 ].split ( '|' )[ 0 ]
                    transcript = row[ 0 ].split ( '|' )[ 1 ]

                    print ( f"Audio file name is {audio_name} &"
                            f" Its Transcription is {transcript}" )
                    if os.path.exists (os.path.join ( audiopath, audio_name ) +".wav"):
                        #audio, sr = sf.read ( os.path.join ( audiopath, audio_name )+".wav" )
                        #sf.write ( file=os.path.join ( outputpath, "wavs2",  audio_name )+".wav", data=audio, samplerate=22050 )
                        #if sr == 22050:
                        fwrite.write(audio_name)
                        fwrite.write("||")
                        fwrite.write ("<"+transcript+">" )
                        fwrite.write("\n")

                        idx += 1

                        if idx == 1000:
                             break
                        # else:
                        #     os.remove(os.path.join ( audiopath, audio_name ) +".wav")
                        #     print(os.path.join ( audiopath, audio_name ) +".wav" + "is removed")

                except:
                    continue
     pass


format_data()