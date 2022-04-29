import os

import torch

# from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig, CharactersConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

output_path = "outputTest"

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
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
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



config = FastPitchConfig(
    run_name="fast_pitch_poland_from_nemo",
    audio=audio_config,
    batch_size=8,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join ( output_path, "f0_cache" ),
    run_eval=False,
    test_delay_epochs=1001,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    use_espeak_phonemes=False,
    print_step=8,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    max_seq_len=500000,
    lr= 0.00001,
    output_path=output_path,
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



model2 = torch.load("/media/ali/G2/codes/NeMo/converted_model.pth")




print(model2['aligner.key_proj.0.conv.bias'].size())

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(restore_path="/media/ali/PrjPrgm/Projects/CoguiTTS/PolandTTS/outputTest/PreTrainModels/tts_models--en--ljspeech--fast_pitch/model_file.pth.tar"),
    config,
    output_path,
    model=model2,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()