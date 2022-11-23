from functools import partial # for auto tuning chain strength
from dwave.embedding.chain_strength import uniform_torque_compensation # for auto tuning chain strength


device = BraketDWaveSampler(device_arn='arn:aws:braket:us-west-2::device/qpu/d-wave/Advantage_system6')
#sampler = neal.SimulatedAnnealingSampler() # if we want to use simulator
sampler = EmbeddingComposite(device)


chain_strength = partial(uniform_torque_compensation, prefactor=2) # this auto-tunes the chain strength
#chain_strength = 200

sampleset = sampler.sample_qubo(qubo, chain_strength=chain_strength, num_reads=1000)


print(sampleset)
