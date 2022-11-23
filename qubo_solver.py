from functools import partial # for auto tuning chain strength
from dwave.embedding.chain_strength import uniform_torque_compensation # for auto tuning chain strength
#from dwave.system import sample
from dwave.system import EmbeddingComposite
import pandas as pd
import pickle
from dwave.system import DWaveSampler
<<<<<<< HEAD
with open('credit_data_qubo.pkl', 'rb') as f:
    qubo = pickle.load(f)
=======
qubo = pd.read_csv('credit_data_qubo.csv', header=None, index=False).to_dict()
>>>>>>> 27a1904de814778b813e5fe7123ceb11f05a8123
# Use a D-Wave system as the sampler
sampler = DWaveSampler() 
print(qubo)

print("QPU {} was selected.".format(sampler.solver.name))

#sampler = neal.SimulatedAnnealingSampler() # if we want to use simulator
embedding_sampler = EmbeddingComposite(sampler)


chain_strength = partial(uniform_torque_compensation, prefactor=2) # this auto-tunes the chain strength
#chain_strength = 200

sampleset = embedding_sampler.sample_qubo(qubo, chain_strength=chain_strength, num_reads=1000)


print(sampleset)
