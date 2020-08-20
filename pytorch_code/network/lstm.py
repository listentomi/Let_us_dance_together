import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import  Variable
import torch.nn.functional as F
import numpy as np
# Device configuration

# Hyper-parameters



# Recurrent neural network (many-to-one)
class SimpleRNN(nn.Module):
    def __init__(self, input_size, fc1_size, hidden_size, num_layers, output_size, device, with_tempo, is_leaky_relu):
        super(SimpleRNN, self).__init__()
        self.acoustic_features = input_size

        self.temporal_features = 3 if with_tempo else 0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.rnn = nn.LSTM(fc1_size , hidden_size, num_layers, batch_first=True, dropout=0.2)
        # self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size+ self.temporal_features, fc1_size)
        self.relu1 = nn.LeakyReLU() if is_leaky_relu else nn.ReLU()
        self.fc = nn.Linear(hidden_size, 80)
        self.relu2 = nn.LeakyReLU() if is_leaky_relu else nn.ReLU()
        self.fc2 = nn.Linear(80, output_size)
        self.device = device

    def forward(self, x):
        x = self.relu1(self.fc1(x))

        # x1 = torch.cat((x,x_temp),dim=2)
        # Set initial hidden and cell states
        h0, c0 = self.init_hidden(x.size(0))

        # Forward propagate LSTM
        out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc2(self.relu2(self.fc(out)))
        return out

    def init_hidden(self, batch_size):
        hidden = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(self.num_layers,batch_size, self.hidden_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()


class LSTM_AE(nn.Module):
    def __init__(self, input_size,output_size,reduced_size,
                 fc1_hidden_size,fc2_hidden_size,fc3_hidden_size,
                 encoder_rnn_hidden_size,decoder_rnn_hidden_size,pred_rnn_hidden_size,
                 num_layers,with_masking ):
        super(LSTM_AE, self).__init__()
        self.acoustic_features = input_size
        self.with_masking = with_masking
        self.temporal_features = 3
        self.hidden_size = encoder_rnn_hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_size, fc1_hidden_size),
            nn.ReLU(),
        )
        self.encoder_rnn = nn.LSTM(fc1_hidden_size + self.temporal_features,encoder_rnn_hidden_size, 1, batch_first=True)
        self.encoder_fc2 = nn.Linear(encoder_rnn_hidden_size, reduced_size)


        self.decoder_fc = nn.Sequential(
            nn.Linear(reduced_size, fc2_hidden_size),
            nn.ReLU(),
        )
        self.decoder_rnn=nn.LSTM(fc2_hidden_size + self.temporal_features, decoder_rnn_hidden_size, 1, batch_first=True)
        self.decoder_fc2 = nn.Linear(decoder_rnn_hidden_size, input_size)

        self.pred_fc = nn.Sequential(
            nn.Linear(reduced_size, fc3_hidden_size),
            nn.ReLU(),
        )
        self.pred_rnn = nn.LSTM(fc3_hidden_size + self.temporal_features, pred_rnn_hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.pred_fc2 = nn.Linear(pred_rnn_hidden_size, output_size )


    def forward(self, x):
        x_ac = x[:,:,:self.acoustic_features]
        x_tempo = x[:,:, -self.temporal_features:]
        beat = x[:,:, -2]
        encoder_fc_out = self.encoder_fc(x_ac)
        encoder_rnn_in = torch.cat((encoder_fc_out, x_tempo), dim=2)
        # Set initial hidden and cell states
        h0, c0 = self.init_hidden(encoder_rnn_in.size(0),1)
        encoder_rnn_out,_ = self.encoder_rnn(encoder_rnn_in,(h0,c0))
        encoder_out = self.encoder_fc2(encoder_rnn_out)

        if(self.with_masking):
            mask = beat # beat frame
            for i in range(encoder_out.shape[-1]):
                encoder_out[:,:,i] = encoder_out[:,:,i].mul_(mask)


        decoder_fc_out = self.decoder_fc(encoder_out)
        decoder_rnn_in = torch.cat((decoder_fc_out, x_tempo), dim=2)
        h1, c1 = self.init_hidden(decoder_rnn_in.size(0),1)
        decoder_rnn_out,_ = self.decoder_rnn(decoder_rnn_in, (h1,c1))
        decoder_out = self.decoder_fc2(decoder_rnn_out)

        pred_fc_out = self.pred_fc(encoder_out)
        pred_rnn_in = torch.cat((pred_fc_out, x_tempo), dim=2)
        h2, c2 = self.init_hidden(pred_rnn_in.size(0),3)
        pred_rnn_out,_ = self.pred_rnn(pred_rnn_in, (h2,c2))
        pred_out = self.pred_fc2(pred_rnn_out)


        return [decoder_out,pred_out]

    def init_hidden(self, batch_size, num_layers):
        hidden = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()

class LSTM_VAE(nn.Module):
    def __init__(self, input_size,output_size,reduced_size,
                 fc1_hidden_size,fc2_hidden_size,fc3_hidden_size,
                 encoder_rnn_hidden_size,decoder_rnn_hidden_size,pred_rnn_hidden_size,
                 num_layers,with_masking ):
        super(LSTM_VAE, self).__init__()
        self.acoustic_features = input_size
        self.with_masking = with_masking
        self.temporal_features = 3
        self.hidden_size = encoder_rnn_hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.z_dim_size=32

        self.vae_encoder=nn.Sequential(
            nn.Linear(
                input_size+self.temporal_features, 
                64
            ),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),

            nn.Linear(
                64, 
                128
            ),
            #nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),

            nn.Linear(
                128, 
                128
            ),
            #nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),
        )

        self.vae_decoder=nn.Sequential(
            nn.Linear(
                27, 
                32
            ),
            #nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),

            nn.Linear(
                32, 
                32
            ),
            #nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),

            nn.Linear(
                32, 
                64
            ),
            #nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),
        )
        self.dense1 = nn.Linear(input_size, input_size)
        self.encoder_rnn = nn.LSTM(self.z_dim_size ,self.z_dim_size, 1, batch_first=True)
        
        self.encoder_fc2 = nn.Linear(self.z_dim_size, reduced_size)

        self.music_out_fc=nn.Linear(64,input_size)

        self.motion_out_fc=nn.Linear(64,output_size)


        self._enc_mu=nn.Linear(128,self.z_dim_size)
        self._enc_logvar=nn.Linear(128,self.z_dim_size)


        self.decoder_fc = nn.Sequential(
            nn.Linear(reduced_size, 24),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),
        )




        self.decoder_rnn=nn.LSTM(fc2_hidden_size + self.temporal_features, decoder_rnn_hidden_size, 1, batch_first=True)



        self.decoder_fc2 = nn.Linear(decoder_rnn_hidden_size, input_size)

        self.pred_fc = nn.Sequential(
            nn.Linear(reduced_size, fc3_hidden_size),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(),
        )
        self.pred_rnn = nn.LSTM(fc3_hidden_size + self.temporal_features, pred_rnn_hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.pred_fc2 = nn.Linear(pred_rnn_hidden_size, output_size )

    def reparameterize(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_logvar(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick

    def forward(self, x):
        x_ac = x[:,:,:self.acoustic_features]
        x_tempo = x[:,:, -self.temporal_features:]
        beat = x[:,:, -2]
        dense1 = self.dense1(x_ac)# 50->50
        vae_encoder_in = torch.cat((dense1, x_tempo), dim=2)  # 50->53

        h_enc=self.vae_encoder(vae_encoder_in)  # 128

        z=self.reparameterize(h_enc) #32

        h0, c0 = self.init_hidden(z.size(0),1)
        encoder_rnn_out,_ = self.encoder_rnn(z,(h0,c0)) # 32->32 


        rnn_out = self.encoder_fc2(encoder_rnn_out)  # 10

        if(self.with_masking):
            mask = beat # beat frame
            for i in range(rnn_out.shape[-1]):
                rnn_out[:,:,i] = rnn_out[:,:,i].mul_(mask)


        decoder_fc_out = self.decoder_fc(rnn_out) # 10->24
        decoder_vae_in = torch.cat((decoder_fc_out, x_tempo), dim=2)# 24->27
        decoder_out = self.vae_decoder(decoder_vae_in) # 24->64
        decoder_out=self.music_out_fc(decoder_out) #64->50



        pred_fc_out = self.pred_fc(rnn_out) # 24
        pred_vae_deocder_in = torch.cat((pred_fc_out, x_tempo), dim=2) #27

        pred_out = self.vae_decoder(pred_vae_deocder_in)# 27->64

        pred_out=self.motion_out_fc(pred_out) # 64->outputsize


        return [decoder_out,pred_out]

    def init_hidden(self, batch_size, num_layers):
        hidden = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(num_layers,batch_size, self.hidden_size), requires_grad=False)
        return hidden.zero_(), cell.zero_()