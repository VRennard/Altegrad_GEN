import time 
import numpy as np
import torch
from torch.nn import Module, ModuleList, Linear,\
        LayerNorm, Sequential, Embedding, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import json
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize

import os

padding_token = '0'
oov_token = '1'
sos_token = '2' # start of sentence, only needed for the target sentence
eos_token = '3' # end of sentence, only needed for the target sentence

from torch.autograd import Variable

class PositionalEncoder(Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x
    
class Transformer(Module):
    """
    """
    ARGS = ["N_stacks_encoder", "N_stacks_decoder", "N_heads",
            "dk", "dv","dmodel", "ff_inner_dim", "vocab_source",
            "vocab_target_inv", "max_size", "device","EOS_token",
            "PAD_token", "SOS_token"]
    def __init__(self, N_stacks_encoder, N_stacks_decoder, N_heads, 
                 dk, dv, dmodel, ff_inner_dim, vocab_source,
                 vocab_target_inv, max_size, device, EOS_token=3, 
                 PAD_token=0, SOS_token=2, OOV_token=1):
        """
        Args:
            N_stacks_encoder (int): Number of encoder stacks.
            N_stacks_decoder (int): Number of encoder stacks.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
            vocab_size_source (int)
            vocab_size_target (int)
            max_size (int): maximum size of sentence.
            device (str): "cpu" or "cuda"
            EOS_token (int): Position of the EOS token in vocabulary.
            PAD_token (int): Position of the PAD token in vocabulary.
            SOS_token (int): Position of the SOS token in vocabulary.
        """
        super(Transformer, self).__init__()
        self.max_size = max_size
        self.EOS_token = EOS_token
        self.SOS_token = SOS_token
        self.PAD_token = PAD_token
        self.OOV_token = OOV_token
        print(SOS_token)
        self.dk = dk
        self.dv = dv
        self.dmodel = dmodel
        self.ff_inner_dim = ff_inner_dim

        self.N_stacks_encoder = N_stacks_encoder
        self.N_stacks_decoder = N_stacks_decoder
        self.N_heads = N_heads

        self.vocab_source = vocab_source
        self.vocab_target_inv = vocab_target_inv
        self.vocab_size_target = len(vocab_target_inv)
        self.source_embedding = Embedding(len(vocab_source)+4, dmodel, PAD_token).to(device)
        self.target_embedding = Embedding(self.vocab_size_target+4, dmodel, PAD_token).to(device)
        self.encoder = Encoder(N_stacks_encoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)
        self.decoder = Decoder(N_stacks_decoder, N_heads, dk, dv, dmodel,
                               ff_inner_dim, device)
        self.output_linear = Linear(dmodel, self.vocab_size_target).to(device)

        #self.PE_tensor = self.build_encoding(max_size, dmodel).to(device)
        self.pe1 = PositionalEncoder(dmodel,self.max_size)
        self.pe2 = PositionalEncoder(dmodel,self.max_size)
        self.device = device

    def forward(self, x, y=None):
        """ Forward pass of the transformer

        Args:
            x (torch.Tensor): Source sentence, tokenized, (B x T)
            y (torch.Tensor): If defined, teacher forcing is used to process
                rapidly the decoding. (B x T_target)

        Returns:
            (B x T x target_vocab_size) the probability distribution over the taret vocab.
        """
        ### TO COMPLETE ###
        src=x
        if self.training and not y is None:
            x=self.source_embedding(x)
            x=self.pe1(x)
            x=self.encoder(x)
            y=self.target_embedding(y)
            y=self.pe2(y)
            x,_=self.decoder(y,x)

            out=self.output_linear(x)
            
            ### TO COMPLETE ###
            
        else :
            outputs = torch.zeros(self.max_size).type_as(src.data)
            outputs[0] = torch.LongTensor([self.SOS_token])
            x=self.source_embedding(x)
            x=self.pe1(x)
            e_outputs=self.encoder(x)
            
            for i in range(1, self.max_size):    
                y=self.target_embedding(outputs[:i].unsqueeze(0))
                y=self.pe2(y)

                o,_=self.decoder(y,e_outputs)

                out=self.output_linear(o)
                val, ix = out[:, -1].data.topk(1)
                outputs[i] = ix[0][0]
                if ix[0][0] == self.EOS_token:
                    break
            y=self.target_embedding(outputs[:i].unsqueeze(0))
            y=self.pe2(y)
            o,_=self.decoder(y,e_outputs)
            out=self.output_linear(o)
                        
          
        return out

    def fit(self, pairs_train, pairs_test, n_epochs, lr=0.001,
            batch_size=64, seed=42, save_path="trained_transformer.pt"):
        """ Trains the network

        Args:
            train_loader (torch.utils.data.DataLoader)
            test_loader (torch.utils.data.DataLoader)
            n_epchos (int)
            lr (float)
            batch_size (int)
            seed (int)
            save_path (str)
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(Dataset(pairs_train), batch_size=batch_size, 
                                  shuffle=True, collate_fn=self.my_pad)
        test_loader = DataLoader(Dataset(pairs_test), batch_size=batch_size, 
                              shuffle=True, collate_fn=self.my_pad)

        optimizer = torch.optim.Adam(self.parameters(), lr)
        criteron = CrossEntropyLoss()

        for epoch in range(n_epochs):
            print(epoch)
            #pbar = self.initialize_pbar(epoch, n_epochs, 136521)
            epoch_train_loss = []
            self.train()
            for i, (source, target) in enumerate(train_loader):
                source = source.to(self.device)
                target = source.to(self.device)

                pred = self.forward(source, target)

                optimizer.zero_grad()
                loss = criteron(pred.flatten(end_dim=1), target.flatten())
                loss.backward()
                optimizer.step()

                #pbar.update(source.size(0))
                epoch_train_loss.append(loss.item())
                #pbar.set_postfix({"train_loss" : np.mean(epoch_train_loss)})

            #test_BLEU = self.test(test_loader, criteron)
            #pbar.set_postfix({"train_loss" : np.mean(epoch_train_loss), "test_BLEU": test_BLEU})
            np.save("epoch_train_loss",epoch_train_loss)    
        #self.save(save_path)

    def test(self, test_loader, criteron):
        """ tests the network on the test_loader 

        Args:
            test_loader (torch.utils.data.DataLoader)
            criteron (torch)
        """
        self.eval()
        BLEU = []
        for i, (source, target) in enumerate(test_loader):
            source = source.to(self.device)
            target = source.to(self.device)
            pred = self.forward(source)
            min_=min(len(pred),len(target))
            BLEU.append(corpus_bleu([str(_) for _ in target.unsqueeze(-1).tolist()[0:min_]],
                                    [str(_) for _ in pred.argmax(-1).tolist()[0:min_] ]))

        return np.mean(BLEU)

    def predict(self, sentence):
        """ Translates a sentence """
        sentence = self.sourceNl_to_ints(sentence)
        pred = self.forward(sentence)
        target_ints = pred.argmax(-1).squeeze() # (<=max_size,1) -> (<=max_size)
        target_nl = self.targetInts_to_nl(target_ints.tolist())

        return ' '.join(target_nl)

    def save(self, path_to_file):
        """ Saves the architecture and weights of the model 

        Args:
            path_to_file (str)
        """
        attrs = {attr : getattr(self, attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs, path_to_file)

    def my_pad(self, my_list):
        """ my_list is a list of tuples of the form [(tensor_s_1,tensor_t_1),...,(tensor_s_batch,tensor_t_batch)]
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence """
        batch_source = pad_sequence([torch.cat((elt[0],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]
        batch_target = pad_sequence([torch.cat((elt[1],torch.LongTensor([self.EOS_token])))
                      for elt in my_list],
                                    batch_first=True,
                                    padding_value=self.PAD_token)[:,:self.max_size-1]

        return batch_source,batch_target 

    def sourceNl_to_ints(self,source_nl):
        '''converts natural language source sentence into source integers'''
        source_nl_clean = source_nl.lower().replace("'",' ').replace('-',' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean, "english")
        source_ints = [int(self.vocab_source[elt]) if elt in self.vocab_source else \
                       self.OOV_token for elt in source_nl_clean_tok] 
        source_ints = torch.LongTensor([source_ints]).to(self.device)

        return source_ints 

    def targetInts_to_nl(self, target_ints):
        '''converts integer target sentence into target natural language'''
        return ['<PAD>' if elt==self.PAD_token else '<OOV>' if elt==self.OOV_token \
                else '<EOS>' if elt==self.EOS_token else '<SOS>' if elt==self.SOS_token\
                else self.vocab_target_inv[elt] for elt in target_ints]

    def shift(self, x):
        """ Adds SOS token to sentence """
        x = torch.cat((torch.ones((x.size(0), 1), dtype=torch.long).to(self.device) * self.SOS_token, x), 1)

        return x

    @classmethod 
    def load(cls, path_to_file, device=None):
        """ Loads a saved model from scrath 

        Args:
            path_to_file (str)
            device (str): If defined, allows to modify to device on which the
                network runs.
        """
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage) 
        state_dict = attrs.pop('state_dict')
        if not device is None:
            attrs["device"] = device

        new = cls(**attrs) 
        new.load_state_dict(state_dict)

        return new

    @staticmethod
    def build_encoding(size, dimension):
        """ Builds a tensor for positional encoding 

        Args:
            size (int): max size of a sentene.
            dimension (int): representation dimentionality in the network.
        """
        pos = torch.arange(size).view(-1,1)
        dim = []
        for _ in range(int(dimension/2)):
            dim.append(np.sin(pos/(10000**(2*_/dimension))))
            dim.append(np.cos(pos/(10000**((2*_+1)/dimension))))

        return torch.cat(dim, 1)

    @staticmethod
    def initialize_pbar(epoch, epochs, its_per_epochs):
        """Initializes a progress bar for the current epoch

        Returns:
            the progess bar (tqdm.tqdm)
        """
        return tqdm(total=its_per_epochs, unit_scale=True,
                    desc="Epoch %i/%i" % (epoch+1, epochs),
                    postfix={})


class Encoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_stacks (int): The number of times the encoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(EncoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Encoder, self).__init__(*stacks)
    def forward(self, x):
        """ Forward pass of the decode
        Needs a modification from the Sequential forward to accept multiple
        inputs.
        """
        for module in self.children():
            x = module(x)

        return x

class EncoderStack(Module):
    """ One stack of encoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the encoder.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(EncoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

    def forward(self, x):
        """ Forward pass of one encoder stack

        Args:
            x (torch.tensor): The input sequence (B, T, dmodel)

        Return:
            (torch.tensor) (B, T, dmodel)
        """
        x =self.attention(x,x,x)
       
        x =self.feed_forward(x)
        return x
    

class Decoder(Sequential):
    def __init__(self, N_stacks, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the decoder.

        Args:
            N_stacks (int): The number of times the decoder is repeated.
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        stacks = []
        for _ in range(N_stacks):
            stacks.append(DecoderStack(N_heads, dk, dv, dmodel, ff_inner_dim,
                                       device))

        super(Decoder, self).__init__(*stacks)

    def forward(self, x, y):
        """ Forward pass of the decode
        Needs a modification from the Sequential forward to accept multiple
        inputs.
        """
        for module in self.children():
            x, y = module(x, y)

        return x, y

class DecoderStack(Module):
    """ One stack of decoder as shown in the original paper """
    def __init__(self, N_heads, dk, dv, dmodel, ff_inner_dim, device):
        """ Initializes the deocder stack.

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
            ff_inner_dim (int): The inner dimensions of feed forward module.
        """
        super(DecoderStack, self).__init__()
        self.dmodel = dmodel
        self.attention_1 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.attention_2 = MultiHeadAttention(N_heads, dk, dv, dmodel, device)
        self.feed_forward = FeedForward(ff_inner_dim, dmodel, device)

    def forward(self, x, y):
        """ Forward pass of one decoder stack
e_outputs
        Args:
            x (torch.tensor): The encoded sequence (B, T, dmodel)
            y (torch.tensor): The shifted decoded sequence (B, T, dmodel)
        """


        x = self.attention_1(x, x, x)
        x =self.attention_2(x, y, y)
        
        x = self.feed_forward(x)
        return x,y



import math
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)  
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(Module):
    """ A MultiHeadAttention Module """
    def __init__(self, N_heads, dk, dv, dmodel, device):
        """ Initializes the Multihead

        Args:
            N_heads (int): The number of attention heads.
            dk (int): The dimension of the space in which attention is computed.
            dv (int): The dimension of the space in which values are sent.
            dmodel (int): The model dimension.
        """
        super(MultiHeadAttention, self).__init__()
        self.dk = dk
        self.N_heads=N_heads
        self.Wq = torch.Tensor(N_heads, dmodel, dk).uniform_(-1, 1).to(device)
        self.Wk = torch.Tensor(N_heads, dmodel, dk).uniform_(-1, 1).to(device)
        self.Wv = torch.Tensor(N_heads, dmodel, dv).uniform_(-1, 1).to(device)
        self.Wo = torch.Tensor(dmodel, dmodel).uniform_(-1, 1).to(device)

    def forward(self, Q, K, V, maskout=False):
        """ forward of a multihead attention

        Specially implemented to be computed in parallel by pytorch.

        Args:
            Q (torch.Tensor): The queries (B, T, dmodel)
            K (torch.Tensor): The keys (B, T, dmodel)
            V (torch.Tensor): The values (B, T, dmodel)
            maskout (bool): Whether to apply or not forward masking.

        Returns:
            (B, T, dmodel) tensor
        """

        outputs = []
        for i in range(self.N_heads):
            q = torch.matmul(Q,self.Wk[i,:,:]) 
            k = torch.matmul(Q,self.Wk[i,:,:]) 
            v = torch.matmul(Q,self.Wk[i,:,:]) 
            outputs.append(attention(q, k, v, self.dk, mask=None, dropout=None))
        result = torch.cat(outputs, dim=-1)
        return torch.matmul(result ,self.Wo)

class FeedForward(Module):
    """ A FeedForward module """
    def __init__(self, inner_dim, dmodel, device):
        """ Initialized the FeedForwrd module

        Args:
            inner_dim (int): Dimension of the hidden layer.
            dmodel (int): The model dimension.
        """
        super(FeedForward, self).__init__()
        self.inner = Linear(dmodel, inner_dim).to(device)
        self.out = Linear(inner_dim, dmodel).to(device)

    def forward(self, x):
        """ Forward of a FeedForward module

        Args:
            x (torch.Tensor)
        """
        x = F.relu(self.inner(x))
        x = self.out(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs) # total nb of observations

    def __getitem__(self, idx):
        source, target = self.pairs[idx] # one observation
        return torch.LongTensor(source), torch.LongTensor(target)

def load_pairs(train_or_test):
    with open("data/" + 'pairs_' + train_or_test + '_ints.txt', 'r', encoding='utf-8') as file:
        pairs_tmp = file.read().splitlines()

    pairs_tmp = [elt.split('\t') for elt in pairs_tmp]
    pairs_tmp = [[[int(eltt) for eltt in elt[0].split()],[int(eltt) for eltt in \
                  elt[1].split()]] for elt in pairs_tmp]

    return pairs_tmp

if __name__ == "__main__":

    pairs_train = load_pairs('train')
    pairs_test = load_pairs('test')

    with open("./data/" + 'vocab_source.json','r') as file:
        vocab_source = json.load(file) # word -> index

    with open("./data/" + 'vocab_target.json','r') as file:
        vocab_target = json.load(file) # word -> index

    vocab_target_inv = {v:k for k,v in vocab_target.items()} # index -> word

    if os.path.exists("trained_transformer.pt"):
        model = Transformer.load("trained_transformer.pt")

    else:
        model = Transformer(N_stacks_encoder=2,
                            N_stacks_decoder=2,
                            N_heads=4, dk=int(256/4), dv=int(256/4), dmodel=256,
                            ff_inner_dim=512,
                            vocab_source=vocab_source,
                            vocab_target_inv=vocab_target_inv,
                            max_size=24, device="cuda", EOS_token=1)
        model.fit(pairs_train, pairs_test, 10, lr=0.001)

    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
               'This river is full of fish.', # plein vs pleine (accord)
               'The fridge is full of food.', 
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace'] 

    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt,model.predict(elt)))
