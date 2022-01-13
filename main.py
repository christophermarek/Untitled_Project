
#local imports
import data_generator

def instruction_generate_dataset(input):
    print('INSTRUCTION: Generate Dataset')

    dataset_title = input[0]
    dataset_size = int(input[1])
    
    dataGenerator = data_generator.DataGenerator(dataset_title)
    output = dataGenerator.generateDataSet(dataset_size)
        
    print(output[1])
    if not output[0]:
        return
    
    print('INSTRUCTION COMPLETE: Generate Dataset \n')

    
    
def main():
    
    print("Program Started \n")
    
    print("Loading Config \n")
    path_to_config = 'config.txt'
    if not path_to_config: 
        print('invalid path to config')
        return
    
    print("Reading Config \n")
    with open(path_to_config) as file:
        
        inComment = False
        inInstruction = False
        capturedLines = list()
        processFunction = None
                
        for line in file:
            if line.strip() == '':
                continue
            
            if line.strip() == '!':
                inComment = not inComment
                continue
            
            if not inComment:
                
                if not inInstruction:
                    # Capture instruction type
                    if line.strip() == 'GENERATEDATASET':
                        processFunction = instruction_generate_dataset
                        inInstruction = True
                else:
                    if line.strip() == 'END':
                        processFunction(capturedLines)
                        inInstruction = False
                        processFunciton = None
                        capturedLines = list()
                    else:
                        capturedLines.append(line.strip())
                    
        
    print("Completed Going Through Config \n")
    
if __name__ == "__main__":
    main()