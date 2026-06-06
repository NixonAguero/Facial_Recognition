from utils import args
from api import verify, register

if __name__ == '__main__':
    args = args.parse_args()

    if args.action == 'sign-up':
        register.register_user(args)
    elif args.action == 'sign-in':
        verify.verify_user(args)
