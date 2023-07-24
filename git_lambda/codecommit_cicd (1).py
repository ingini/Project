import os
import json
import boto3
import zipfile
from pyathena import connect 

base_dir = '/tmp'

repository_name='REPO NAME '
zipfile_name = 'lambda_function.zip'
temporary_bucket = 'BUCKET NAME '
temporary_key = 'lambda_deploy/lambda_function.zip'

os.chdir(base_dir)

codecommit_client = boto3.client('codecommit', region_name='ap-northeast-2')
lambda_client = boto3.client('lambda', region_name='ap-northeast-2')
s3_client = boto3.client('s3', region_name='ap-northeast-2')


def getLastCommitID(repository, branch="master"):
    response = codecommit_client.get_branch(
        repositoryName=repository,
        branchName=branch
    )
    commitId = response['branch']['commitId']
    return commitId

def getLastCommitLog(repository, commitId):
    response = codecommit_client.get_commit(
        repositoryName=repository,
        commitId=commitId
    )
    return response['commit']

def getFileDifferences(repository_name, lastCommitID, previousCommitID):
    response = None
    if previousCommitID != None:
        response = codecommit_client.get_differences(
            repositoryName=repository_name,
            beforeCommitSpecifier=previousCommitID,
            afterCommitSpecifier=lastCommitID
        )
    else:
        # The case of getting initial commit (Without beforeCommitSpecifier)
        response = codecommit_client.get_differences(
            repositoryName=repository_name,
            afterCommitSpecifier=lastCommitID
        )
    differences = []
    if response == None:
        return differences
    while "nextToken" in response:
        response = codecommit_client.get_differences(
            repositoryName=repository_name,
            beforeCommitSpecifier=previousCommitID,
            afterCommitSpecifier=lastCommitID,
            nextToken=response["nextToken"]
        )
        differences += response.get("differences", [])
    else:
        differences += response["differences"]
    return differences

def update_function(codecommit_folder_path):
    function_name = os.path.basename(codecommit_folder_path)

    response = codecommit_client.get_folder(
        repositoryName=repository_name,
        folderPath=codecommit_folder_path
    )

    if 'files' in response:
        if os.path.exists(zipfile_name):
            os.remove(zipfile_name)
        with zipfile.ZipFile(zipfile_name, 'a') as zf:
            for file in response['files']:
                file_path = file['absolutePath']

                response = codecommit_client.get_file(
                    repositoryName=repository_name,
                    filePath=file_path
                )

                download_path = os.path.basename(file_path)

                with open(download_path, 'wb') as f:
                    file_content = response['fileContent']
                    f.write(file_content)

                zf.write(download_path)
                os.remove(download_path)

    s3_client.upload_file(zipfile_name, temporary_bucket, temporary_key)

    response = lambda_client.update_function_code(
        FunctionName=function_name,
        S3Bucket=temporary_bucket,
        S3Key=temporary_key
    )

    s3_client.delete_object(Bucket=temporary_bucket, Key=temporary_key)

def lambda_handler(event, context):    
    branch = 'master'
        
    last_commit_id = getLastCommitID(repository_name, branch)
    last_commit = getLastCommitLog(repository_name, last_commit_id)
    prev_commit_id = None
    if len(last_commit['parents']) > 0:
        prev_commit_id = last_commit['parents'][0]
    
    differences = getFileDifferences(repository_name, last_commit_id, prev_commit_id)

    deployed_functions = []

    for difference in differences:
        if 'afterBlob' in difference:
            download_file = difference['afterBlob']['path']

            if download_file.startswith('lambda'):
                codecommit_folder_path = os.path.dirname(download_file)
                if not codecommit_folder_path in deployed_functions:
                    update_function(codecommit_folder_path)
                    deployed_functions.append(codecommit_folder_path)


    return {
        'statusCode': 200,
        'body': json.dumps('success')
    }

if __name__ == '__main__':
    lambda_handler(None, None)
