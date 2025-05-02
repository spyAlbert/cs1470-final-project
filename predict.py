import clip
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import argparse
import skimage.io as io
import PIL.Image
from clip_caption import ClipCaptionModel, ClipCaptionPrefix


def _get_initial_embedding(model, tokenizer, prompt_text=None, prompt_tokens=None, embed=None):
    device = next(model.parameters()).device
    if embed is not None:
        return embed, None

    if prompt_tokens is None:
        ids = tokenizer.encode(prompt_text)
        tokens_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    else:
        tokens_tensor = prompt_tokens.to(device)

    embedded = model.gpt.transformer.wte(tokens_tensor)
    return embedded, tokens_tensor


def _filter_logits_with_top_p(logits, p_threshold):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    to_remove = cumulative_probs > p_threshold
    to_remove[..., 1:] = to_remove[..., :-1].clone()  
    to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[to_remove]
    logits[:, indices_to_remove] = -float("inf")
    return logits


def _generate_single_sample(
    model, tokenizer, embed_input, token_input, max_len, stop_id, top_p, temperature
):
    current_embed = embed_input
    current_tokens = token_input
    device = current_embed.device

    for _ in range(max_len):
        output = model.gpt(inputs_embeds=current_embed)
        logits = output.logits[:, -1, :] / max(temperature, 1e-5)
        logits = _filter_logits_with_top_p(logits, top_p)

        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        next_embed = model.gpt.transformer.wte(next_token)

        if current_tokens is None:
            current_tokens = next_token
        else:
            current_tokens = torch.cat((current_tokens, next_token), dim=1)

        current_embed = torch.cat((current_embed, next_embed), dim=1)

        if next_token.item() == stop_id:
            break

    return current_tokens


def generate_with_prompt(
    model,
    tokenizer,
    prompt_tokens=None,
    prompt_text=None,
    embed=None,
    num_outputs=1,
    max_tokens=67,
    top_p=0.8,
    temperature=1.0,
    stop_token=".",
):
    model.eval()
    results = []
    stop_token_id = tokenizer.encode(stop_token)[0]

    with torch.no_grad():
        for _ in range(num_outputs):
            embed_input, token_input = _get_initial_embedding(
                model, tokenizer, prompt_text, prompt_tokens, embed
            )
            full_token_sequence = _generate_single_sample(
                model,
                tokenizer,
                embed_input,
                token_input,
                max_tokens,
                stop_token_id,
                top_p,
                temperature,
            )
            decoded = tokenizer.decode(full_token_sequence.squeeze().cpu().tolist())
            results.append(decoded)

    return results[0]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_type',type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--image', default='COCO_val2014_000000114147.jpg')
    parser.add_argument('--model', default='coco_weights_mlp.pt')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # load Caption model
    if args.mapping_type == 'mlp':
        prefix_length = 10
        model = ClipCaptionModel(prefix_length=prefix_length,clip_len=10,transformer_layers=8,projection_method=args.mapping_type)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
  
    else:
        prefix_length = 40
        model = ClipCaptionPrefix(prefix_length=prefix_length,clip_len=40,transformer_layers=8,projection_method=args.mapping_type)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))

    model = model.eval().to(device)
    
    # load image
    image_path = args.image
    image = io.imread(image_path)
    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)

    # extract feature
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    # generate caption
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    caption = generate_with_prompt(model, tokenizer, embed=prefix_embed)

    # out
    print(caption)

if __name__ == "__main__":
    main()