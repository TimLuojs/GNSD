import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from dlutils import *
from const import *
torch.manual_seed(1)
import copy

class GNSD(nn.Module):
	def __init__(self, feats):
		super(GNSD, self).__init__()
		self.name = 'GNSD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(
			nn.Linear(2 * feats, feats),
			nn.ReLU(),
			nn.Linear(feats, feats),
			nn.ReLU(),
			nn.Linear(feats, feats),
			nn.Sigmoid()
		)
		self.save_decoder1 = None # for recovery from adv train 
	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2
	def unfreeze_layers_decoder(self, decoder_to_train):
		assert decoder_to_train in ['decoder1', 'decoder2'],"decoder_to_train must be 'decoder1' or 'decoder2'."
		for param in self.parameters():
			param.requires_grad = False

		if decoder_to_train == 'decoder1':
			for param in self.transformer_decoder1.parameters():
				param.requires_grad = True
		elif decoder_to_train == 'decoder2':
			for param in self.transformer_decoder2.parameters():
				param.requires_grad = True
	def unfreeze_all(self):
		for param in self.parameters():
			param.requires_grad = True

class GNSDSingleStation(nn.Module):
	def __init__(self, feats):
		super(GNSDSingleStation, self).__init__()
		self.name = 'GNSDSingle'
		self.lr = lr
		self.batch = 128
		feats = int(feats/5)
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(
			nn.Linear(2 * feats, feats),
			nn.ReLU(),
			nn.Linear(feats, feats),
			nn.ReLU(),
			nn.Linear(feats, feats),
			nn.Sigmoid()
		)
		self.save_decoder1 = None # for recovery from adv train 
	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		src = src[:,:,5:10]
		tgt_mid = tgt[:,:,5:10]
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt_mid)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt_mid)))
		tgt1 = copy.deepcopy(tgt)
		tgt2 = copy.deepcopy(tgt)
		tgt1[:,:,5:10] = x1
		tgt2[:,:,5:10] = x2 
		return tgt1, tgt2
	def unfreeze_layers_decoder(self, decoder_to_train):
		assert decoder_to_train in ['decoder1', 'decoder2'],"decoder_to_train must be 'decoder1' or 'decoder2'."
		for param in self.parameters():
			param.requires_grad = False

		if decoder_to_train == 'decoder1':
			for param in self.transformer_decoder1.parameters():
				param.requires_grad = True
		elif decoder_to_train == 'decoder2':
			for param in self.transformer_decoder2.parameters():
				param.requires_grad = True
	def unfreeze_all(self):
		for param in self.parameters():
			param.requires_grad = True

