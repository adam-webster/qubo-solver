from functools import partial # for auto tuning chain strength
from dwave.embedding.chain_strength import uniform_torque_compensation # for auto tuning chain strength
from dwave.system import sample_qubo
from dwave.system import EmbeddingComposite
import pandas as pd

from dwave.system import DWaveSampler
qubo = pd.read_csv('credit_data_qubo.csv', header=None).to_dict()
# Use a D-Wave system as the sampler
sampler = DWaveSampler() 

print("QPU {} was selected.".format(sampler.solver.name))

#sampler = neal.SimulatedAnnealingSampler() # if we want to use simulator
embedding_sampler = EmbeddingComposite(sampler)


chain_strength = partial(uniform_torque_compensation, prefactor=2) # this auto-tunes the chain strength
#chain_strength = 200

sampleset = embedding_sampler.sample_qubo(qubo, chain_strength=chain_strength, num_reads=1000)


print(sampleset)
