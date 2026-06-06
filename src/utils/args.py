import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Facial Recognition System')
    
    parser.add_argument('--action', type=str, required=True, choices=['sign-up', 'sign-in'], default='sign-in', help='Action to perform: sign-up or sign-in')
    parser.add_argument('--method', type=str, required=True, choices=['hybrid', 'standard'], default='standard', help='Recognition method to use')
    
    
    return parser.parse_args()