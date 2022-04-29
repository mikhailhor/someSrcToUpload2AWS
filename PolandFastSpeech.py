import os

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager
from TTS.tts.configs.shared_configs import CharactersConfig


# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.txt",
    # meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt"),
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)) , "Prepared"),
)

audio_config = BaseAudioConfig(
    sample_rate=16000,
    do_trim_silence=True,
    trim_db=60,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

character_config = CharactersConfig(
    pad=" ",
    bos=">",
    eos="<",
    characters="aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż",
    punctuations=None,
    unique=True
)



config = FastSpeechConfig(
    run_name="fast_speech_polish",
    audio=audio_config,
    batch_size=16,
    compute_input_seq_cache=True,
    compute_f0=False,
    run_eval=False,
    lr=0.0001,
    test_delay_epochs=40001,
    epochs=40000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    use_espeak_phonemes=False,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    max_seq_len=500000,
    output_path="outputTest",
    datasets=[dataset_config],
    test_sentences=[
       "Tak",
        "Nie",
        "Cześć",
        "Dzień dobry",
        "Dobry wieczór",
        "Jak masz na imię?"
    ],
    characters=character_config,
    enable_eos_bos_chars=True
)

# compute alignments
if not config.model_args.use_aligner:
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
    # TODO: make compute_attention python callable
    os.system(
        f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile metadata.csv --data_path ./recipes/ljspeech/LJSpeech-1.1/  --use_cuda true"
    )

# init audio processor
ap = AudioProcessor(**config.audio)

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init the model
model = ForwardTTS(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(restore_path="/media/ali/PrjPrgm/Projects/CoguiTTS/KurdishTTS/outputTest/fast_speech_kurdish-February-15-2022_03+11AM-c63bb481/checkpoint_110000.pth.tar"),
    config,
    "outputTest",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
