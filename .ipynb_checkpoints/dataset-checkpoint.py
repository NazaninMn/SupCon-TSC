

import torch
import numpy as np


class Dataset_HandMovementDirection:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'right':
            current_target = 0
        elif current_target == 'left':
            current_target = 1
        elif current_target == 'forward':
            current_target = 2
        else:
            current_target = 3

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_ArticularyWordRecognition:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]      

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}


class Dataset_JapaneseVowels:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]      

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}    
    
class Dataset_AtrialFibrillation:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'n':
            current_target = 0
        elif current_target == 's':
            current_target = 1
        else:
            current_target = 2

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
# data class
class Dataset_BasicMotions:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'Standing':
            current_target = 0
        elif current_target == 'Running':
            current_target = 1
        elif current_target == 'Badminton':
            current_target = 2
        else:
            current_target = 3

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_CharacterTrajectories:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_FaceDetection:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_Heartbeat:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'normal':
            current_target = 1
        else:
            current_target = 0

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_MotorImagery:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'finger':
            current_target = 0
        else:
            current_target = 1

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_NATOPS:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]


        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}


class Dataset_PEMS_SF:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]


        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_PenDigits:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]


        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_StandWalkJump:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'standing':
            current_target = 0
        elif current_target == 'walking':
            current_target = 1
        else:
            current_target = 2

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_SpokenArabicDigits:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_PenDigits:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
    
class Dataset_Cricket:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_EigenWorms:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}


class Dataset_ERing:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_Handwriting:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
    
class Dataset_Libras:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_UWaveGestureLibrary:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
    
class Dataset_LSST:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 6:
            current_target = 0
        elif current_target == 15:
            current_target = 1
        elif current_target == 16:
            current_target = 2
        elif current_target == 42:
            current_target = 3
        elif current_target == 52:
            current_target = 4
        elif current_target == 53:
            current_target = 5
        elif current_target == 62:
            current_target = 6
        elif current_target == 64:
            current_target = 7
        elif current_target == 65:
            current_target = 8
        elif current_target == 67:
            current_target = 9
        elif current_target == 88:
            current_target = 10
        elif current_target == 90:
            current_target = 11
        elif current_target == 92:
            current_target = 12
        elif current_target == 95:
            current_target = 13

           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    

class Dataset_SelfRegulation_SCP2:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'positivity':
            current_target = 1
        else:
            current_target = 0

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_SelfRegulation_SCP1:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'positivity':
            current_target = 1
        elif current_target == 'negativity':
            current_target = 0

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
    
class Dataset_MotorImagery:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'finger':
            current_target = 1
        else:
            current_target = 0

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}

    
class Dataset_DuckDuckGeese:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'Black-bellied_Whistling_Duck':
            current_target = 0
        elif current_target == 'Canadian_Goose':
            current_target = 1
        elif current_target == 'Greylag_Goose':
            current_target = 2
        elif current_target == 'Pink-footed_Goose':
            current_target = 3
        elif current_target == 'White-faced_Whistling_Duck':
            current_target = 4

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_RacketSports:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'Badminton_Clear':
            current_target = 0
        elif current_target == 'Badminton_Smash':
            current_target = 1
        elif current_target == 'Squash_BackhandBoast':
            current_target = 2
        else:
            current_target = 3

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_Epilepsy:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'EPILEPSY':
            current_target = 0
        elif current_target == 'RUNNING':
            current_target = 1
        elif current_target == 'SAWING':
            current_target = 2
        else:
            current_target = 3

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_FingerMovements:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'right':
            current_target = 0
        elif current_target == 'left':
            current_target = 1

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
class Dataset_EthanolConcentration:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'E35':
            current_target = 0
        elif current_target == 'E38':
            current_target = 1
        elif current_target == 'E40':
            current_target = 2
        elif current_target == 'E45':
            current_target = 3

        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}

class Dataset_InsectWingbeat:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if current_target == 'aedes_female':
            current_target = 0
        elif current_target == 'aedes_male':
            current_target = 1
        elif current_target == 'fruit_flies':
            current_target = 2
        elif current_target == 'house_flies':
            current_target = 3
        elif current_target == 'quinx_female':
            current_target = 4    
        elif current_target == 'quinx_male':
            current_target = 5
        elif current_target == 'stigma_female':
            current_target = 6  
        elif current_target == 'stigma_male':
            current_target = 7     
        elif current_target == 'tarsalis_female':
            current_target = 8 
        elif current_target == 'tarsalis_male':
            current_target = 9     
            
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}    
    
class Dataset_Phoneme:
    def __init__(self, x_train_np, y_train_np, transforms):
        self.data = x_train_np
        self.targets = y_train_np
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]

        if current_target == 'AA':
            current_target = 0
        elif current_target == 'AE':
            current_target = 1
        elif current_target == 'AH':
            current_target = 2
        elif current_target == 'AO':
            current_target = 3
        elif current_target == 'AW':
            current_target = 4
        elif current_target == 'AY':
            current_target = 5
        elif current_target == 'B':
            current_target = 6
        elif current_target == 'CH':
            current_target = 7
        elif current_target == 'D':
            current_target = 8
        elif current_target == 'DH':
            current_target = 9
        elif current_target == 'EH':
            current_target = 10
        elif current_target == 'ER':
            current_target = 11
        elif current_target == 'EY':
            current_target = 12
        elif current_target == 'F':
            current_target = 13
        elif current_target == 'G':
            current_target = 14
        elif current_target == 'HH':
            current_target = 15
        elif current_target == 'IH':
            current_target = 16    
        elif current_target == 'IY':
            current_target = 17
        elif current_target == 'JH':
            current_target = 18  
        elif current_target == 'K':
            current_target = 19
        elif current_target == 'L':
            current_target = 20  
        elif current_target == 'M':
            current_target = 21
        elif current_target == 'N':
            current_target = 22  
        elif current_target == 'NG':
            current_target = 23
        elif current_target == 'OW':
            current_target = 24  
        elif current_target == 'OY':
            current_target = 25
        elif current_target == 'P':
            current_target = 26  
        elif current_target == 'R':
            current_target = 27
        elif current_target == 'S':
            current_target = 28  
        elif current_target == 'SH':
            current_target = 29
        elif current_target == 'T':
            current_target = 30  
        elif current_target == 'TH':
            current_target = 31  
        elif current_target == 'UH':
            current_target = 32
        elif current_target == 'UW':
            current_target = 33  
        elif current_target == 'V':
            current_target = 34
        elif current_target == 'W':
            current_target = 35    
        elif current_target == 'Z':
            current_target = 36  
        elif current_target == 'ZH':
            current_target = 37              
        else:
            current_target = 38
           
        if self.transform:
            current_sample = self.transform(current_sample.astype(np.float64))
            current_target = torch.tensor(current_target, dtype=torch.long)

        return {"x": current_sample,
                "y": current_target}
    
    


def Select_Dataset(name):
    if 'HandMovementDirection' in name:
        return Dataset_HandMovementDirection
    elif 'ArticularyWordRecognition' in name:
        return Dataset_ArticularyWordRecognition
    elif 'AtrialFibrillation' in name:
        return Dataset_AtrialFibrillation
    elif 'BasicMotions' in name:
        return Dataset_BasicMotions
    elif 'CharacterTrajectories' in name:
        return Dataset_CharacterTrajectories
    elif 'FaceDetection' in name:
        return Dataset_FaceDetection
    elif 'Heartbeat' in name:
        return Dataset_Heartbeat
    elif 'NATOPS' in name:
        return Dataset_NATOPS
    elif 'MotorImagery' in name:
        return Dataset_MotorImagery
    elif 'PEMS-SF' in name:
        return Dataset_PEMS_SF
    elif 'PenDigits' in name:
        return Dataset_PenDigits
    elif 'StandWalkJump' in name:
        return Dataset_StandWalkJump
    elif 'SpokenArabicDigits' in name:
        return Dataset_SpokenArabicDigits
    elif 'SelfRegulation_SCP2' in name:
        return Dataset_SelfRegulation_SCP2
    elif 'Phoneme' in name:
        return Dataset_Phoneme
    elif 'Cricket' in name:
        return Dataset_Cricket
    elif 'DuckDuckGeese' in name:
        return Dataset_DuckDuckGeese
    elif 'EigenWorms' in name:
        return Dataset_EigenWorms
    elif 'Epilepsy' in name:
        return Dataset_Epilepsy
    elif 'EthanolConcentration' in name:
        return Dataset_EthanolConcentration
    elif 'ERing' in name:
        return Dataset_ERing
    elif 'Handwriting' in name:
        return Dataset_Handwriting
    elif 'Libras' in name:
        return Dataset_Libras
    elif 'LSST' in name:
        return Dataset_LSST
    elif 'Libras' in name:
        return Dataset_Libras
    elif 'MotorImagery' in name:
        return Dataset_MotorImagery
    elif 'RacketSports' in name:
        return Dataset_RacketSports
    elif 'SelfRegulationSCP1' in name:
        return Dataset_SelfRegulation_SCP1
    elif 'UWaveGestureLibrary' in name:
        return Dataset_UWaveGestureLibrary
    elif 'FingerMovements' in name:
        return Dataset_FingerMovements
    elif 'JapaneseVowels' in name:
        return Dataset_JapaneseVowels
    elif 'InsectWingbeat' in name:
        return Dataset_InsectWingbeat
    
    
    