import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pathlib import Path
from language.vocabulary import Vocabulary 
import pandas as pd

def runDataSetPrep():
    # unDataSetPrep is the "main" interface that lets you execute all the 
    # functions. 
    #
    #
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()


def printEncodedTokens():
    vocab = Vocabulary.load('language/vocabulary.pkl')
    utterances = ['there should not be a shade', 'remove the shade', 'the shade is not needed', 'the shade is not necessary', 'the shade is not wanted', 'the shade is not desired', 'get rid of the shade', 'the shade does not exist', 'there is no light covering']
    tokens = [['there', 'should', 'not', 'be', 'a', 'shade'], ['remove', 'the', 'shade'], ['the', 'shade', 'is', 'not', 'needed'], ['the', 'shade', 'is', 'not', 'necessary'], ['the', 'shade', 'is', 'not', 'wanted'], ['the', 'shade', 'is', 'not', 'desired'], ['get', 'rid', 'of', 'the', 'shade'], ['the', 'shade', 'does', 'not', 'exist'], ['there', 'is', 'no', 'light', 'covering']]
    # loop over utterances
    for i, utterance in enumerate(utterances):
        print(f'Utterance: {utterance}')
        print("i", i)
        print("tokens: ", tokens[i])
        token_encoded_chosen = vocab.encode(tokens[i], max_len=18)
        print("encoded tokens: ", token_encoded_chosen)

def genPCNoise():
    from gaussianhelper import add_gaussian_noise_to_point_cloud
    from visualization import plot_3d_point_cloud
    # get the folders in the pcinput directory
    class_folders = os.listdir('pcinput')
    num_class_folders = len(class_folders)
    for i, fn in enumerate(class_folders):
        count_contents = 0
        if fn == '.DS_Store':
            continue
        print("i", i)
        print("fn", fn)
        class_folder_path = os.path.join('pcinput', fn)
        folder_contents = os.listdir(class_folder_path)
        # check the number of files in each folder
        num_files = len(folder_contents)
        # check if specific file name in contents 
        if '.DS_Store' in folder_contents:
            print("DS_Store exists in folder: ", fn)
        #many = np.unique(folder_contents)
        print(fn, "num_files", num_files)
        #print("wait how many?", many)
        for j, fn2 in enumerate(folder_contents):
            if fn2 == '.DS_Store':
                continue
            print("fn2", fn2)
            count_contents += 1
            print("count_contents", count_contents)
            file_path = os.path.join(class_folder_path, fn2)
            print("file_path", file_path)
            #1st load npz file
            data = np.load(file_path, allow_pickle=True)
            #print(data.files)
            # Load the npy file from the npz file
            point_cloud = data['pointcloud']
            #plot_3d_point_cloud(point_cloud, figsize=(10, 10), set_lim=0.2, visualization_pc_axis=(0,2,1))
            #sigma of .01 is pretty decent for a noisy point cloud, still fairly recognizable 
            sigmas = [.003, .005, .008, .01, .015, .02, .025, .03, .035, .04, .045, .05, .055, .06, .065, .07, .075, .08, .085, .09, .1, .2]
            for sigma in sigmas:
                new_noisy_pc = add_gaussian_noise_to_point_cloud(point_cloud, sigma)
                #save the noisy point cloud in same npz format as the original
                name_wo_ext = fn2.split('.npz')[0]
                save_path = f'100pcPerClassOutput/{fn}/sigma_{sigma}/noisy_pc_{name_wo_ext}_sigma_{sigma}.npz'
                np.savez(save_path, pointcloud=new_noisy_pc)
                #plot_3d_point_cloud(new_noisy_pc, figsize=(10, 10), set_lim=0.2, visualization_pc_axis=(0,2,1))
                print(f"Noisy Point Cloud Done for sigma: {sigma}")
                
def countContents():
    class_folders = os.listdir('pcinput')
    num_class_folders = len(class_folders)
    for i, fn in enumerate(class_folders):
        count_contents = 0
        if fn == '.DS_Store':
            continue
        print("i", i)
        print("fn", fn)
        class_folder_path = os.path.join('pcinput', fn)
        folder_contents = os.listdir(class_folder_path)
        # check the number of files in each folder
        num_files = len(folder_contents)
        # check if specific file name in contents 
        if '.DS_Store' in folder_contents:
            print("DS_Store exists in folder: ", fn)
        #many = np.unique(folder_contents)
        print(fn, "num_files", num_files)
        #print("wait how many?", many)
        for j, fn2 in enumerate(folder_contents):
            if fn2 == '.DS_Store':
                continue
            count_contents += 1
        print("count_contents", count_contents)

def newCSVsForNoise():
    df = pd.read_csv('data/shapetalk_preprocessed_public_version_0.csv')
    # limit this to test data for changeit_split jk had to add in other scenarios for val and train due to lack of unique files
    df = df[df['changeit_split'] != 'ignore']
    df_test = df[df['changeit_split'] == 'test']
    df_val = df[df['changeit_split'] == 'val']
    df_train = df[df['changeit_split'] == 'train']
    # print the length of the dataframe
    print("Length of the dataframe after filtering: ", len(df))
    
    # get the folders in the pcinput directory
    class_folders = os.listdir('pcinput')
    num_class_folders = len(class_folders)
    for i, fn in enumerate(class_folders):
        if fn == '.DS_Store':
            continue
        class_folder_path = os.path.join('pcinput', fn)
        folder_contents = os.listdir(class_folder_path)
        rows = []
        for j, fn2 in enumerate(folder_contents):
            if fn2 == '.DS_Store':
                continue
            name_no_ext = fn2.split('.npz')[0]
            
            data = df_test[df['source_model_name'] == f'{name_no_ext}']
            if len(data) == 0:
                data = df_val[df['source_model_name'] == f'{name_no_ext}']
                if len(data) == 0:
                    data = df_train[df['source_model_name'] == f'{name_no_ext}']
                    if len(data) == 0:
                        print("No data found for: ", name_no_ext, " in ", fn)
                        continue
            rows.append(data.iloc[[0]])
        # create a new dataframe with the rows
        new_df = pd.concat(rows)
        # if changeit_split column value is not 'test' then update it to 'test'
        new_df['changeit_split'] = 'test'
        # if source_unary_split column value is not 'test' then update it to 'test'
        new_df['source_unary_split'] = 'test'
        fileNamesToGenerate = ['baseline_exp1', 'noiseSigma003_exp1', 'noiseSigma005_exp1', 'noiseSigma008_exp1', 'noiseSigma01_exp1', 'noiseSigma015_exp1', 'noiseSigma02_exp1', 'noiseSigma025_exp1', 'noiseSigma03_exp1', 'noiseSigma035_exp1', 'noiseSigma04_exp1', 'noiseSigma045_exp1', 'noiseSigma05_exp1', 'noiseSigma055_exp1', 'noiseSigma06_exp1', 'noiseSigma065_exp1', 'noiseSigma07_exp1', 'noiseSigma075_exp1', 'noiseSigma08_exp1', 'noiseSigma085_exp1', 'noiseSigma09_exp1', 'noiseSigma1_exp1', 'noiseSigma2_exp1']
        sigmas = [.003, .005, .008, .01, .015, .02, .025, .03, .035, .04, .045, .05, .055, .06, .065, .07, .075, .08, .085, .09, .1, .2]
        # loop over length of fileNamesToGenerate
        for i, fileName in enumerate(fileNamesToGenerate):
            temp_df = new_df.copy()
            if i == 0:
                temp_df.to_csv(f'output/{fn}_{fileName}.csv', index=False)
            else:
                temp_df['source_model_name'] = temp_df['source_model_name'].apply(lambda x: f'noisy_pc_{x}_sigma_{sigmas[i-1]}')
                temp_df['source_uid'] = temp_df['source_uid'].apply(lambda x: f'100pcPerClassOutput/{fn}/sigma_{sigmas[i-1]}/noisy_pc_{x.split("/")[-1]}_sigma_{sigmas[i-1]}.npz')
                temp_df.to_csv(f'output/{fn}_{fileName}.csv', index=False)
    return

def newCSVsForLanguageManipulation():
    vocab = Vocabulary.load('language/vocabulary.pkl')
    df = pd.read_csv('data/shapetalk_preprocessed_public_version_0.csv')
    # limit this to test data for changeit_split
    df = df[df['changeit_split'] == 'test']
    df_vase = df[df['source_object_class'] == 'vase']
    df_lamp = df[df['source_object_class'] == 'lamp']
    df_chair = df[df['source_object_class'] == 'chair']
    print("Length of the vase dataframe after filtering: ", len(df_vase))
    print("Length of the lamp dataframe after filtering: ", len(df_lamp))
    print("Length of the chair dataframe after filtering: ", len(df_chair))
    
    use_dfs = [df_vase, df_lamp, df_chair]
    for i, the_df in enumerate(use_dfs):
        # set up baseline dataset by choosing 20 rows that include one of these words
        words = ['fat', 'thin', 'short', 'tall', 'wide', 'narrow', 'big', 'small', 'large', 'tiny', 'huge', 'little', 'massive', 'skinny', 'gigantic', 'petite', 'enormous', 'immense', 'miniscule', 'vast', 'circular', 'comfortable', 'soft', 'shorter', 'taller', 'girthy', 'sharp', 'round', 'bent', 'broader', 'narrower', 'curvy', 'flat', 'longer', 'shorter', 'straight', 'bumpy', 'smooth']
        # get 20 random rows from the dataframe if the utterance of the row does not contain any of the words in the words list choose another row
        random_rows = the_df[the_df['utterance'].str.contains('|'.join(words))].sample(n=20)
        #random_rows = the_df.sample(n=20)
        baseline_df = pd.DataFrame(random_rows)
        baseline_df.to_csv(f'output/{baseline_df.iloc[0]["source_object_class"]}_baseline.csv', index=False)
        #setup new dataframe for the modified baseline that we will edit data for each row
        modified_baseline_df = pd.DataFrame(columns=baseline_df.columns)
        # now get each row of the baseline dataset and edit the utterance and tokens, then  add it as a line to the modified_baseline_df
        for j, row in baseline_df.iterrows():
            utterance = row['utterance']
            # create a gui to display the utterance and allow the user to change it
            # show the original utterance
            print("Original Utterance: ", utterance)
            # get the new utterance
            new_utterance = input("Enter the new utterance: ")
            #set up tokens
            tokens = new_utterance.split(' ')
            print("Tokens: ", tokens)
            # get the length of the tokens 
            tokens_len = len(tokens)
            print("Tokens Length: ", tokens_len)
            # encode the tokens
            tokens_encoded = vocab.encode(tokens, max_len=18)
            print("Encoded Tokens: ", tokens_encoded)
            # add the new row to the modified_baseline_df
            modified_baseline_df = modified_baseline_df.append({'workerid': row['workerid'], 'utterance': new_utterance, 'assignmentid': row['assignmentid'], 'worktimeinseconds': row['worktimeinseconds'], 'source_model_name': row['source_model_name'], 'source_object_class': row['source_object_class'], 'source_dataset': row['source_dataset'], 'target_model_name': row['target_model_name'], 'target_object_class': row['target_object_class'], 'target_dataset': row['target_dataset'], 'is_patched': row['is_patched'], 'target_uid': row['target_uid'], 'source_uid': row['source_uid'], 'hard_context': row['hard_context'], 'target_original_object_class': row['target_original_object_class'], 'source_original_object_class': row['source_original_object_class'], 'saliency': row['saliency'], 'tokens': tokens, 'tokens_len': tokens_len, 'utterance_spelled': new_utterance, 'target_unary_split': row['target_unary_split'], 'source_unary_split': row['source_unary_split'], 'listening_split': row['listening_split'], 'changeit_split': row['changeit_split'], 'tokens_encoded': tokens_encoded}, ignore_index=True)
        modified_baseline_df.to_csv(f'output/{baseline_df.iloc[0]["source_object_class"]}_modified_language.csv', index=False)


'''def genCSVWithRemoval():
    # Define the CSV file name and headers
    csv_file = 'output/custom_utterances_remove_shade.csv'
    headers = ['workerid', 'utterance', 'assignmentid', 'worktimeinseconds', 'source_model_name',
            'source_object_class', 'source_dataset', 'target_model_name', 'target_object_class',
            'target_dataset', 'is_patched', 'target_uid', 'source_uid', 'hard_context',
            'target_original_object_class', 'source_original_object_class', 'saliency', 'tokens',
            'tokens_len', 'utterance_spelled', 'target_unary_split', 'source_unary_split',
            'listening_split', 'changeit_split', 'tokens_encoded']
    vocab = Vocabulary.load('language/vocabulary.pkl')

    # Open the CSV file in write mode and write headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        # prep array of utterances
        utterances = ['there should not be a shade', 'remove the shade', 'the shade is not needed', 'the shade is not necessary', 'the shade is not wanted', 'the shade is not desired', 'get rid of the shade', 'the shade does not exist']
        tokens = [['there', 'should', 'not', 'be', 'a', 'shade'], ['remove', 'the', 'shade'], ['the', 'shade', 'is', 'not', 'needed'], ['the', 'shade', 'is', 'not', 'necessary'], ['the', 'shade', 'is', 'not', 'wanted'], ['the', 'shade', 'is', 'not', 'desired'], ['get', 'rid', 'of', 'the', 'shade'], ['the', 'shade', 'does', 'not', 'exist']]
        default_saliency = 0
        # Loop through pcoutput folder to generate dummy data for each noisy point cloud
        class_folders = Path('./pcoutput')
        for filepath in class_folders.iterdir():
            print("filepath: ", filepath)
            if filepath.name == '.DS_Store':
                    continue
            object_class = filepath.name
            print("object_class: ", object_class)
            class_folder_path = os.path.join(class_folders, object_class)
            #print("class_folder_path: ", class_folder_path)
            folder_contents = os.listdir(class_folder_path)
            #print("folder_contents: ", folder_contents)
            for j, fn2 in enumerate(folder_contents):
                if fn2 == '.DS_Store':
                    continue
                file_path = os.path.join(class_folder_path, fn2)
                #print("file path: ", file_path )
                og_id = fn2.split('noisy_pc_')[1]
                og_id = og_id.split('.npz')[0]
                noisy_id = fn2.split('.npz')[0] + '.npz' + fn2.split('.npz')[1]
                source_model_name = og_id
                target_model_name = noisy_id
                target_uid = f'pcinput/{object_class}/{og_id}' # e.g. filepath of the OG point cloud
                #print("Target uid:", target_uid)
                source_uid = os.path.join(class_folder_path, noisy_id) # e.g. filepath of the noisy point cloud
                #print("Source uid:", source_uid)
                #if source_model_name contains the source object class, then ModelNet, else ShapeNet
                if object_class in og_id:
                    targetdataset = 'ModelNet'
                else:
                    targetdataset = 'ShapeNet'
                #print(targetdataset)
                #get the length of my utterance list 
                len_utterances = len(utterances)
                #generate a random number within the length of the utterances list
                rand_num = np.random.randint(0, len_utterances)
                chosen_utterance = utterances[rand_num]
                token_chosen = tokens[rand_num]
                #max_len = max(len(s) for s in token_chosen)
                #token_encoded_chosen = token_chosen.apply(lambda x: vocab.encode_token(x, max_len=max_len))
                token_encoded_chosen = vocab.encode(token_chosen, max_len=18)
                #print(rand_num)
                # Generate dummy data for each iteration (replace with your actual data generation logic)
                row_data = [
                    f'user_9999',
                    f'{chosen_utterance}',
                    f'assignment_{j}',
                    f'{200}',
                    f'{source_model_name}',
                    f'{object_class}',
                    f'CustomNoiseyDataset',
                    f'{target_model_name}',
                    f'{object_class}',
                    f'{targetdataset}',
                    f'FALSE',
                    f'{target_uid}',
                    f'{source_uid}',
                    f'TRUE',
                    f'{object_class}',
                    f'{object_class}',
                    f'{default_saliency}',
                    f'{token_chosen}',
                    f'{len(token_chosen)}',
                    f'{chosen_utterance}',
                    f'test',
                    f'test',
                    f'test',
                    f'test',
                    f'{token_encoded_chosen}',
                ]

                # Write generated data to the CSV file
                writer.writerow(row_data)

        print(f'CSV file "{csv_file}" created successfully with generated data!')


    #Visualize the original and noisy point clouds
    #plot_3d_point_cloud(point_cloud, figsize=(10, 10), set_lim=0.2, visualization_pc_axis=(0,2,1))
    #plot_3d_point_cloud(new_noisy_pc, figsize=(10, 10), set_lim=0.2, visualization_pc_axis=(0,2,1))
    
    #save the noisy point cloud in same npz format as the original
    #np.savez('pcoutput/noisy_pc_1c73c8ff52c31667c8724d5673a063a6.npz', pointcloud=new_noisy_pc)
    #test_2_data = np.load('pcoutput/noisy_pc_1c73c8ff52c31667c8724d5673a063a6.npz')
    #print(test_2_data.files)
    #test2pcdata = test_2_data['pointcloud']
    #print("Noisy Point Cloud Done")
    #plot_3d_point_cloud(test2pcdata, figsize=(10, 10), set_lim=0.2, visualization_pc_axis=(0,2,1))
'''

def displayPC():
    from visualization import plot_3d_point_cloud
    curclass='lamp'
    original = np.load(f'pcinput/{curclass}/771d4def2e44bc169eb34048e600e1ea.npz')
    # does 100pcPerClassOutput/{curclass}/sigma_0.003/ exist check if it does
    filepath = f'100pcPerClassOutput/{curclass}/sigma_0.003/noisy_pc_vase_0255_sigma_0.003.npz'
    folder = Path(f'100pcPerClassOutput/{curclass}/sigma_0.003')
    # check if the folder exists
    if os.path.exists(folder):
        print("Folder exists")
    else:
        print("Folder does not exist")
    if os.path.exists(filepath):
        print("File exists")
    else:
        print("File does not exist")
    sigma_003 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.003/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.003.npz')
    sigma_005 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.005/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.005.npz')
    sigma_008 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.008/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.008.npz')
    sigma_01 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.01/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.01.npz')
    sigma_015 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.015/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.015.npz')
    sigma_02 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.02/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.02.npz')
    sigma_025 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.025/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.025.npz')
    sigma_03 = np.load(f'100pcPerClassOutput/{curclass}/sigma_0.03/noisy_pc_771d4def2e44bc169eb34048e600e1ea_sigma_0.03.npz')
    
    mySet = [original, sigma_003, sigma_005, sigma_008, sigma_01, sigma_015, sigma_02, sigma_025, sigma_03]
    titleAry = ["Original Point Cloud", "Noisy Point Cloud Sigma: 0.003", "Noisy Point Cloud Sigma: 0.005", "Noisy Point Cloud Sigma: 0.008", "Noisy Point Cloud Sigma: 0.01", "Noisy Point Cloud Sigma: 0.015", "Noisy Point Cloud Sigma: 0.02", "Noisy Point Cloud Sigma: 0.025", "Noisy Point Cloud Sigma: 0.03"]
    
    for i, data in enumerate(mySet):
        title = titleAry[i]
        pcdata = data['pointcloud']
        plot_3d_point_cloud(pcdata, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=title)

def displayPCsFromCustNpy():
    from visualization import plot_3d_point_cloud
    languageFolder = 'generation_results/language/use/'
    noisyFolder = 'generation_results/noisy_point_clouds/'
    
    # get the files in the folder directory
    languageFiles = os.listdir(languageFolder)
    print("languageFiles: ", languageFiles)
    noisyResultFiles = os.listdir(noisyFolder)
    
    # loop over the files languageFiles
    for i, fn in enumerate(languageFiles):
        if fn == '.DS_Store':
            continue
        print("i", i)
        print("fn", fn)
        # if fn contains vase_baseline
        if 'vase_baseline' in fn:
            # pull relevant files to display source and target 
            # use pd to read csv file 
            csvfile = 'output/vase_baseline.csv'
            continue
        elif 'lamp_baseline' in fn:
            # pull relevant files to display source and target
            csvfile = 'output/lamp_baseline.csv'
            # quit to go to next file/next loop
            continue
        elif 'chair_baseline' in fn:
            # pull relevant files to display source and target
            csvfile = 'output/chair_baseline.csv'
            continue
        elif 'vase_modified_lang' in fn:
            # pull relevant files to display source and target
            csvfile = 'output/vase_modified_language.csv'
            continue
        elif 'lamp_modified_lang' in fn:
            # pull relevant files to display source and target
            csvfile = 'output/lamp_modified_language.csv'
        elif 'chair_modified_lang' in fn:
            # pull relevant files to display source and target
            csvfile = 'output/chair_modified_language.csv'
            continue
        else:
            print("No match found")
        #
        # read the csv file
        df = pd.read_csv(csvfile)
        # get the source_uid and target_uid for every row in the dataframe and push to an array 
        source_uids = df['source_uid']
        target_uids = df['target_uid']
        utterances = df['utterance']
        # load the full npy contents of the file as a dataframe
        data = np.load(f'{languageFolder}{fn}', allow_pickle=True).item()
        reconstructed = data['recons']
        print("reconstructed length", len(reconstructed))
        print("reconstructed 0 ", reconstructed[0].shape)
        print("reconstructed 1", reconstructed[1].shape)
        # is reconstructed[0] equal to reconstructed[1] if so print "Equal"
        if np.array_equal(reconstructed[0], reconstructed[1]):
            print("Equal")
        else:
            print("Not Equal")
        # plot the reconstructed point clouds
        # iterate over reconstructed[0]'s 0th index to get the point cloud
        #samplesourcefile = 'pointclouds/lamp/ShapeNet/cbe4301ef1418bd8b036d6b8e2579386.npz'
        #sampletargetfile = 'pointclouds/lamp/ShapeNet/5d265be0ec25af3a7f772a7ed7ffb61.npz'
        
        #samplesource = np.load(samplesourcefile, allow_pickle=True)
        #sampletarget = np.load(sampletargetfile, allow_pickle=True)
        
        #source = samplesource['pointcloud']
        #target = sampletarget['pointcloud']
        
        print("File reviewing: ", fn)
        for j, pc in enumerate(reconstructed[0]):
            if source_uids[j] != 'chair/ShapeNet/6e1e73e14637a28da1c367d7a459a9b7.npz': 
                # need to concat pointclouds/ with the source_uid to get a filepath
                source_pc = np.load(f'pointclouds/{source_uids[j]}.npz', allow_pickle=True)
                source = source_pc['pointcloud']
                target_pc = np.load(f'pointclouds/{target_uids[j]}.npz', allow_pickle=True)
                target = target_pc['pointcloud']
                print("Utterance: ", utterances[j])
                plot_3d_point_cloud(source, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Source Point Cloud')
                plot_3d_point_cloud(target, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Target Point Cloud')
                plot_3d_point_cloud(pc, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Resulting Point Cloud')
            
def displayPCsFromNoisyNpy():
    from visualization import plot_3d_point_cloud
    noisyFolder = 'generation_results/noisy_point_clouds/'
    
    #check if folder exists
    if os.path.exists(noisyFolder):
        print("Folder exists")
    
    rand_num = np.random.randint(1, 100)
    
    # get the files in the folder directory
    noisyResultFiles = os.listdir(noisyFolder)
    print ("noisyResultFiles: ", noisyResultFiles)
    
    # loop over the files languageFiles
    for i, fn in enumerate(noisyResultFiles):
        if fn == '.DS_Store':
            continue
        print("i", i)
        print("fn", fn)
        # if fn contains vase_baseline
        if 'lamp' not in fn:
            continue
        elif 'lamp_baseline' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_baseline.csv'
            continue
        elif '0.003_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma003_exp1.csv'
        elif '0.005_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma005_exp1.csv'
        elif '0.008_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma008_exp1.csv'
        elif '0.01_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma01_exp1.csv'
        elif '0.015_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma015_exp1.csv'
        elif '0.02_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma02_exp1.csv'
        elif '0.025_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma025_exp1.csv'
        elif '0.03_lamp' in fn:
            # pull relevant files to display source and target
            csvfile = f'lampnoisecsvs/lamp_noiseSigma03_exp1.csv'
        else:
            print("No match found")
            
        # check if csv file exists
        if os.path.exists(csvfile):
            print("CSV file exists")
            df = pd.read_csv(csvfile)
            # get the source_uid and target_uid for every row in the dataframe and push to an array 
            source_uids = df['source_uid']
            target_uids = df['target_uid']
            utterances = df['utterance']
            # load the full npy contents of the file as a dataframe
            data = np.load(f'{noisyFolder}{fn}', allow_pickle=True).item()
            reconstructed = data['recons']
            print("File reviewing: ", fn)
            
            pc = reconstructed[0][rand_num]
            # need to concat pointclouds/ with the source_uid to get a filepath
            source_pc = np.load(f'{source_uids[rand_num]}', allow_pickle=True)
            source = source_pc['pointcloud']
            target_pc = np.load(f'pointclouds/{target_uids[rand_num]}.npz', allow_pickle=True)
            target = target_pc['pointcloud']
            print("Utterance: ", utterances[rand_num])
            plot_3d_point_cloud(source, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Source Point Cloud')
            plot_3d_point_cloud(target, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Target Point Cloud')
            plot_3d_point_cloud(pc, figsize=(8, 8), set_lim=0.2, visualization_pc_axis=(0,2,1), title=f'Resulting Point Cloud')
        else:
            print("CSV file does not exist")
            
        
    
if __name__ == '__main__':
    newCSVsForLanguageManipulation()
    genPCNoise()
    newCSVsForNoise()
    #printEncodedTokens()
    #countContents()
    #displayPCsFromNoisyNpy()
    #displayPCsFromCustNpy()
    #displayPC()