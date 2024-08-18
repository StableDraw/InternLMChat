import torch
from os.path import isfile, exists
from os import mkdir
from transformers import AutoTokenizer, AutoModelForCausalLM

'''
prompt = "diving"
segments_duration = 8

response, history = model.chat(tokenizer, f"I'm creating a clip from a neural network that generates video. The theme of my video is \"f{prompt}\". The clip consists of f{segments_duration} second segments. Please write prompts for the neural network that would describe each segment of this clip. Describe each segment separately. Write each prompt from the next line and no additional information", history=[])
print(response)

# Hello! How can I help you today?

print(response)
'''



def reset_history(user_id: str):
    '''
    Функция, обновляющий беседу с пользователем
    Принимает строковый id пользователя
    '''

    with open("users_history\\" + user_id + ".txt", "w"): #Очищаем содержимое файла пользователя
        pass



def get_history(user_id: str) -> list:
    '''
    Функция дял получения истории переписки пользователя
    Принимает строковый id пользователя
    Возвращает список запросов к нейронке в виде списка строк (если запросов не было - возвращает пустой список)
    '''

    history = []
        
    with open("users_history\\" + user_id + ".txt", "r") as f:
        while True:
            line = f.readline() #Считываем строку из файла
            if not line: #Если она пуста - выходим
                break
            line = line.replace("\\n", "\n")
            line = line[:-1] if line[-1] == "\n" else line 
            history.append(line) #Записываем декодированную строку в список

    return history



def save_history(history: list, user_id: str): 
    '''
    Функция для сохранения истории переписок пользователя
    Принимает историю переписки в виде списка строк и строковый id пользователя
    '''

    with open("users_history\\" + user_id + ".txt", "w") as f:
        f.writelines(str(item).replace("\n", "\\n") + "\n" for item in history) #построчно записывам историю переписки в файл



def history_to_nn_history(history: list):
    '''
    Функция для конфертации списка истории в историю, понятную нейронке (список кортежей)
    Принимает список истории запросов
    '''

    nn_history = [] #Список для хранения истории нейросети

    for i in range(0, len(history), 2):
        nn_history.append((history[i], history[i + 1]))

    return nn_history



class InternLMChat():
    '''
    Чат-бот: нейронная сеть, способная генерировать текстовые ответы на текстовые сообщения пользователя, учитывая контекст беседы
    '''

    def __init__(self):
        '''
        Инициализация класса нейронки для начала общения
        '''
        
        if not exists("users_history"):
            mkdir("users_history")

        model_path = "weights"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)

        # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, trust_remote_code = True).cuda()
        self.model = self.model.eval()


    def do_chat(self, prompt: str, user_id: str = None) -> str:
        '''
        Метод, генерирующий текстовый ответ на текстовый запрос от пользователя, поддерживая беседу
        Принимает текст и строковый id идентификатор пользователя
        Возвращает текст
        '''

        history = get_history(user_id = user_id) #Получаем историю переписок этого пользователя

        nn_history = history_to_nn_history(history = history) #Конвертируем историю в понятный нейронной сети формат

        response, new_history = self.model.chat(self.tokenizer, prompt, history = nn_history) #Генерируем ответ на запрос

        history.append(new_history[-1][0]) #Добавляем в общую историю запрос от пользователя
        history.append(new_history[-1][1]) #Добавляем в общую историю ответ от нейронной сети

        torch.cuda.empty_cache() #Очищаем видеопамять

        save_history(history = history, user_id = user_id) #Запомининаем историю пользователя

        return response


    def text_to_text(self, prompt: str) -> str:
        '''
        Метод, генерирующий текстовый ответ на единственный текстовый запрос
        Принимает текст
        Возвращает текст
        '''

        response, _ = self.model.chat(self.tokenizer, prompt, history = []) #Генерируем ответ на запрос

        torch.cuda.empty_cache() #Очищаем видеопамять

        return response



if __name__ == "__main__":

    cb = InternLMChat()

    user_id = "0"

    reset_history(user_id = user_id)

    prompt1 = "Hello, how are you?"

    response1 = cb.do_chat(prompt = prompt1, user_id = user_id)

    print("response 1:", response1)

    prompt2 = "What do you think about weather today?"

    response2 = cb.do_chat(prompt = prompt2, user_id = user_id)

    print("response 2:", response2 + "\n")

    history = get_history(user_id = user_id)

    for msg in history:
        print(msg)

    reset_history(user_id = user_id)

    print("reset_history\n")

    history = get_history(user_id = user_id)

    for msg in history:
        print(msg)

    print("new conversation:\n")

    prompt3 = "Hello, remember this numbers for me: \"734\""

    response3 = cb.do_chat(prompt = prompt3, user_id = user_id)

    print("responce 3:", response3)

    prompt4 = "Please tell me the numbers that I asked you to remember"

    response4 = cb.do_chat(prompt = prompt4, user_id = user_id)

    print("responce 4:", response4 + "\n" + "new conversation 2:")

    reset_history(user_id = user_id)

    prompt5 = "Hello, remember this numbers for me: \"734\""

    response5 = cb.do_chat(prompt = prompt5, user_id = user_id)

    print("response 5:", response5)

    prompt6 = "Please tell me the numbers that I asked you to remember"

    response6 = cb.text_to_text(prompt = prompt6)

    print("responce 6:", response6)