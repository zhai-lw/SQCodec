from functools import lru_cache

import torch

import sq_codec

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def eval_model(codec: sq_codec.SQCodec, audio_in: torch.Tensor):
    codec.network.to(device=DEVICE)
    audio_in = audio_in.to(device=DEVICE)

    codec.network.eval()
    with torch.inference_mode():
        audio_batch, audio_length = audio_in.shape
        q_feature, indices = codec.encode_audio(audio_in)
        audio_out = codec.decode_audio(q_feature)  # or
        # audio_out = codec.decode_audio(indices=indices)
        generated_audio = audio_out[:, :audio_length].detach()

    return ((audio_in - generated_audio) ** 2).mean().item()


@lru_cache
def example_audio(target_sample_rate=16000):
    import librosa
    sample_audio, sample_rate = librosa.load(librosa.example("libri1"))
    sample_audio = sample_audio[None, :]
    print(f"loaded sample audio and audio sample_rate :{sample_rate}")
    sample_audio = librosa.resample(sample_audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    print(f"resampled sample audio to target sr:{target_sample_rate}")
    return sample_audio


def main():
    all_models = sq_codec.list_models()
    print(f"Available models: {all_models}")

    for model_name in all_models:
        codec = sq_codec.get_model(model_name)
        audio_in = example_audio(codec.config.sample_rate)
        model_mse = eval_model(codec, torch.tensor(audio_in))
        print(f"{model_name} mse:{model_mse}")


if __name__ == '__main__':
    main()
