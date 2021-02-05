import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        #initialize weights
        self.init_weights()        
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)               
    
    def init_hidden_weights(self, batch_size):
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        return torch.zeros(1, batch_size, self.hidden_size).to(device), torch.zeros(1, batch_size, self.hidden_size).to(device)       
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeds = self.word_embeddings(captions)
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden_weights(self.batch_size)

        features = features.unsqueeze(1)
        inputs = torch.cat((features,embeds), dim=1)
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outputs = self.linear(lstm_out)
        return outputs     
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        preds = []
        count = 0
        word_item = None
        
        while count < max_len and word_item != 1 :
            
            #Predict output
            lstm_out, states = self.lstm(inputs, states)
            output = self.linear(lstm_out)
            
            #Get max value
            prob, word = output.max(2)
            
            #append word
            word_item = word.item()
            preds.append(word_item)
            
            #next input is current prediction
            inputs = self.word_embeddings(word)
            
            count+=1
        
        return preds