from utils.consts import Consts


class Particle():
    def __init__(self,name,mass, charge, lattice, energy):
        self.name = name
        self.mass = mass* 1e6
        self.charge = charge
        self.lattice = lattice
        self.energy = energy



class PreCreatedParticles():
    Ar_CM1 = Particle('Ar_CM1', mass=37218.59218810, charge=12, lattice={Consts.SOL: [0.81, -0.81, -0.83, -0.89, -0.75]},energy=1.356*40)

