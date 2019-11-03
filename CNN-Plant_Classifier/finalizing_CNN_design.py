import pandas as pd
import os

max_accuracy_avg = 0
pathname = 'Hyperparameter_tuning/'

def find_parameters(val_acc_values,acc_values,parameter):

	global max_accuracy_avg
	for i,j,z in zip(val_acc_values,acc_values,parameter):
		print("*"*30)
		avg = (i+j)/2.0
		print("avg= {} and max_avg = {} ".format(avg,max_accuracy_avg),end="-"*10+'>')
		print("the parameters is : {} ".format(z))
		if max_accuracy_avg <= avg:
			max_accuracy_avg = avg
			final_parameters = z
		else:
			continue

	return final_parameters




def final_params():
	dir_lis = os.listdir(pathname)[1:]
	print(dir_lis)
	for name in dir_lis:
		file_values = pd.read_csv(pathname+"/"+name)
		val_acc_values = file_values.iloc[:,2].values
		acc_values = file_values.iloc[:,4].values
		parameters = file_values.iloc[:,5:].values
		final_parameters = find_parameters(val_acc_values,acc_values,parameters)
	final_params_dict = {'activation':final_parameters[0],'batch_size':final_parameters[1],'dense_neuron':final_parameters[2],'dropout':final_parameters[3],'optimizer':final_parameters[4]}
	return final_params_dict

#
# if __name__ == "__main__":
# 	params = final_params()