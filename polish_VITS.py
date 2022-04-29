import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.txt",
    # meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt"),
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)) , "Prepared"),
)

audio_config = BaseAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=False,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

character_config = CharactersConfig(
    pad=" ",
    bos=">",
    eos="<",
    characters="aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż",
    punctuations="'"
)


# This is the config that is saved for the future use
config = VitsConfig(
    run_name="VITS_polish",
    run_description="Fine Tune VITS to polish Sentences",
    audio=audio_config,
    batch_size=8,
    run_eval=False,
    test_delay_epochs=1001,
    r=1,
    lr_gen=0.00001,
    lr_disc=0.00001,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    use_espeak_phonemes=False,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path="outputTest",
    max_seq_len=500000,
    datasets=[dataset_config],
    test_sentences=[[
       "Tak",
        "Nie",
        "Cześć",
        "Dzień dobry",
        "Dobry wieczór",
        "Jak masz na imię?"
    ]],
    characters=character_config,
    enable_eos_bos_chars=True

)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, evalSamples = load_tts_samples(dataset_config, eval_split=True, formatter=None)

# init model
model = Vits(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(restore_path="outputTest/PreTrainModels/tts_models--en--ljspeech--vits/model_file.pth.tar"),
    config,
    output_path="outputTest",
    model=model,
    train_samples=train_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()


