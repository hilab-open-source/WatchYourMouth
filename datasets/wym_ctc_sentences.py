import torch
import numpy as np
from torch.utils.data import Dataset
# from torch.utils.data.dataloader import default_collate, DataLoader
import tqdm
class MouthActionSentence3D(Dataset):
    def __init__(self, root, dataset, num_points = 1024, padding=True):
        super(MouthActionSentence3D).__init__()
        self.videos = []
        self.texts = []
        self.users = []
        self.pos = []
        self.folder = dataset
        self.num_points = num_points
        self.letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 
                   'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
                   'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 
                   'X', 'Y', 'Z']
        # self.words = [' ', 'AND', 'OR', 'THAT', 'IF', 'LIKE', 'WHO', 'WHAT',
        #               'ME', 'YOU', 'IT', 'MUSIC', 'ALARM', 'VOLUME',
        #               'MESSAGE', 'WEATHER', 'PLAY', 'SWITCH', 'CONTINUE',
        #               'SET', 'LISTEN', 'EVERY', 'ANOTHER', 'ALL', 'SOME',
        #               'THIS', 'POPULAR', 'FAST', 'HAPPY', 'UPCOMING', 'WARM',
        #               'FROM', 'ABOUT', 'BETWEEN', 'UNTIL', 'AFTER', 'NEARBY',
        #               'HERE', 'WHEN', 'BACK', 'WHY']
        self.padding = padding
        with open(root+dataset) as file:
            video_name =  file.read().splitlines()
        # print("loading data ...")
        for v in tqdm.tqdm(video_name):
            if not padding:
                v.replace("Point_padding", "Point_zip")
            video = np.load(root+'Sentences/'+v, allow_pickle=True)
            try:
                video = np.array([video[k] for k in video], dtype=object)
            except Exception as e:
                print(e, v)
                continue
            # shorten the video length
            video = [v for i, v in enumerate(video) if i%4 != 0]
            self.videos.append(video)

            text = v.split('/')[-1].split('.')[0]
            self.texts.append(text)

            position = v.split('/')[0].split('_')[-1]
            self.pos.append(position)

            user = v.split('/')[0].split('_')[-2]
            if len(user) == 5:
                user = int(user[-1:])
            else:
                user = int(user[-2:])
            self.users.append(user)

    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        video = self.videos[index]
        text = self.texts[index]
        user = self.users[index]
        position = self.pos[index]

        video = [video[i] for i in range(len(video))]
        for i, p in enumerate(video):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            video[i] = p[r, :]

        video = np.array(video)
        
        text = [self.letters.index(c)+1 for c in text.upper()]
        
        return video, text, video.shape[0], len(text), int(user), position
    
# if __name__ == '__main__':
#     root = '../UserStudy2.0/'
#     dataset = 'Sentences/Train3.txt'
#     datasets = MouthActionSentence3D(root=root, dataset=dataset)
#     video, text, video_len, text_len, user = datasets[1]