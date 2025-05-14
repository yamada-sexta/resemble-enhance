import argparse
import gradio as gr
import torch
import torchaudio

from resemble_enhance.enhancer.inference import denoise, enhance


def _fn(path, solver, nfe, tau, denoising):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    return (new_sr, wav1), (new_sr, wav2)


def main():
    parser = argparse.ArgumentParser(description="Resemble Enhance Audio Processing")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to use for processing")

    args = parser.parse_args()

    global device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = [
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver"),
        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature"),
        gr.Checkbox(value=False, label="Denoise Before Enhancement"),
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]

    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
        inputs=inputs,
        outputs=outputs,
    )

    interface.launch(share=args.share)


if __name__ == "__main__":
    main()
