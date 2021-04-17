import librosa
import numpy as np

import config as cfg

def get_wav(filename, sr=cfg.sr):
  src1_src2 = librosa.load(filename, sr=sr, mono=False)[0]
  mixed = librosa.to_mono(src1_src2)
  src1, src2 = src1_src2[0, :], src1_src2[1, :]
  return mixed, src1, src2

def to_wav_file(mag, phase, len_hop=cfg.hop_size):
  stft_maxrix = get_stft_matrix(mag, phase)
  return np.array(librosa.istft(stft_maxrix.T, hop_length=len_hop))

def to_spec(wav, len_frame=cfg.frame_size, len_hop=cfg.hop_size):
  return librosa.stft(np.require(wav, requirements="F"), n_fft=len_frame, hop_length=len_hop).T

def get_stft_matrix(magnitudes, phases):
  return magnitudes * np.exp(1.j * phases)

def rndshift(x, max_offset):
  l = x.shape[0]
  assert max_offset > 0
  assert l > max_offset
  
  offset = np.random.randint(0, max_offset)
  return x[offset:-max_offset + offset]

def infer(x, n_out_channel, train, ver=None, gene=None):
  import sys
  if not "../lib" in sys.path:
    sys.path.append("../lib")
  import geneop
  import config as cfg
  if ver is not None:
    geneop.load_type(ver)
    if isinstance(gene, int):
      gene = geneop.cvtintlst(gene, geneop.gene_len)
    else:
      raise TypeError("Invalid gene value `%r`" % (gene,))
  else:
    geneop.load_type(cfg.gene_ver)
    gene = geneop.cvtintlst(cfg.gene_value, geneop.gene_len)
  return geneop.build_from_gene(x, n_out_channel, gene)
