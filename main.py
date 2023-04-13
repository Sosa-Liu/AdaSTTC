import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np

quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)
tip.automf(3, 'quant')

rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'], 'R1')
rule2 = ctrl.Rule(service['average'], tip['average'], 'R2')
rule3 = ctrl.Rule(quality['good'] | service['good'] , tip['high'], 'R3')
# rule4 = ctrl.Rule(quality['average'], tip['average'], 'R4') 

system = ctrl.ControlSystem([rule1, rule2, rule3])
sys = ctrl.ControlSystemSimulation(system)



