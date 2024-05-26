# import some packages you need here
import torch

def generate(model, seed_characters, temperature, data, max_length, device, *args):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """
    
    # write your codes here
    hidden = model.init_hidden(1)
    # 숨겨진 상태와 셀 상태를 각각 디바이스로 이동
    if isinstance(hidden, tuple):  # LSTM의 경우
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:  # RNN의 경우
        hidden = hidden.to(device)

    char_dict = data.char_index
    dict_char = data.index_char
    input_tensor = torch.tensor([[char_dict[char] for char in seed_characters]]).to(device)

    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output = output/temperature
            output_probs = torch.softmax(output[-1].squeeze(), dim=0)           
            predict_char_index = torch.multinomial(output_probs, num_samples=1).item()
            predict_index = torch.tensor([[predict_char_index]]).cuda()
            predicted_char = dict_char[predict_char_index]
            input_tensor = torch.cat([input_tensor, predict_index], dim=1)
            seed_characters += predicted_char
    sample = seed_characters

    return sample