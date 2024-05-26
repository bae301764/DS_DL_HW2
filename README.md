Report File

# 1. Network structure

RNN 모델과 LSTM 모델은 각각 다음과 같은 구조로 구성되어 있다.

|RNN|LSTM|input size|hidden size|
|---|---|----|---|
|Embedding layer|Embedding layer|주어진 data에 포함된 character의 unique개수|128|
|RNN layer (num_layers=2)|LSTM layer (num_layers=2)|128|256|
|Dense layer1|Dense layer1|256|64|
|GELU|GELU|||
|Dense layer2|Dense layer2|64|16|unique
|GELU|GELU|||
|Dense layer3|Dense layer3|16|주어진 data에 포함된 character의 unique개수|


# 2. Experimental Setup

learning rate = 0.001\
weight_decay = 0.001\
Optimizer = AdamW\
epoch = 250\
batch size = 256\
validation rate = 0.2\
random seed = 5\
validation loss가 가장 낮을 때의 weight을 저장

# 3. Compare Model (RNN, LSTM)

그래프를 통해 각 모델의 validation loss를 확인했을 때 LSTM의 loss가 RNN보다 작은 것을 확인할 수 있고, 이는 LSTM이 RNN보다 language generation 성능이 더 좋다고 볼 수 있다.
<img src="https://github.com/bae301764/DS_DL_HW2/blob/main/Compare%20Language%20Model's%20Loss.png">


위쪽은 LSTM의 loss 그래프, 아래는 RNN의 loss 그래프이다.
<img src="https://github.com/bae301764/DS_DL_HW2/blob/main/LSTM's%20Loss.png">
<img src="https://github.com/bae301764/DS_DL_HW2/blob/main/RNN's%20Loss.png">
# 4. Text generate example

아래 seed_sentence는 문장을 생성하고자 만든 예시 문장이다.

seed_sentence = ["First Citizen: O, fleeting time",
                "All: In twilight's grasp, I fi",
                "Second Citizen: Whispered drea",
                "MENENIUS: Upon yon hills, our ",
                "MARCIUS: Stars above, guide me"]

RNN generate.txt과 LSTM generate.txt는 위의 예시문을 각각 RNN과 LSTM에 입력으로 넣었을 때 100자로 생성한 문장이다.

Softmax의 parameter인 temperature를 [0.1, 0.5, 0.7, 1, 2]로 바꿔주면서 temperature에 따라 생성문장이 어떻게 달라지는지 확인하였다.

generation 성능이 더 좋은 LSTM을 기준으로 확인했을 때 다음과 같다.





Temperature가 낮은 경우
---
첫번째\
First Citizen: O, fleeting times,
Unto a lineal true-derived course.

Lord Mayor:
Do, good my lord, your citizens entreat you.

BUC

두번째\
All: In twilight's grasp, I find love so barr'd,
Which renders good for bad, blessings for curses.

LADY ANNE:
Villain, thou know'

세번째\
Second Citizen: Whispered dream, what is to be dreaded.

SICINIUS:
Tell not me:
I know this cannot be.

BRUTUS:
Not possible.

Mes

네번째\
MENENIUS: Upon yon hills, our knees,
Which we disdain should tatter us, yet sought
The very way to catch them.

BRUTUS:
You speak 

다섯번째\
MARCIUS: Stars above, guide me by invention, I
Will answer in mine honour.

MENENIUS:
Ay, but mildly.

CORIOLANUS:
Well then, I pr



Temperature가 높은 경우
---
첫번째\
First Citizen: O, fleeting times to nekent of the Juciours and to suver for me! the air.

QUEEN ELIZABETH:
The heaven, I this man?

두번째\
All: In twilight's grasp, I fisrect the malice and bereaves the state
Of that I un's person; his apel's word proceed adped you wit

세번째\
Second Citizen: Whispered dream, you shall bear the justice, the people were not whether
He that he come.

Second Servingman:
What

네번째\
MENENIUS: Upon yon hills, our wives, and children, power and perceive
He did so set his
teeth and tear it; O, I wash we have sent-

다섯번째\
MARCIUS: Stars above, guide me 't will be dangerous to be the bed'st.
Breaking him too, I metwine and succeters, flest: no pegind 

-> Temperature를 낮게 설정하면 단어들의 다양성이 낮아져 \n 와 같은 단어들이 반복적으로 나타나는 것을 볼 수 있음.\
   Temperature를 높게 설정할 경우 단어들의 다양성이 높아져 다양한 철자들을 선택하기 때문에 문장의 형태가 길어지고 반복적인 단어가 나타나기 어려움.
