from nacl.signing import VerifyKey
from eth_account.messages import encode_defunct
from eth_account import Account
from api.api_types import SolanaVerifyRequest, EvmVerifyRequest
from base58 import b58decode
from server.firebase import auth


def _firebase_custom_token(uid: str) -> str:
    custom_token = auth.create_custom_token(uid)

    if isinstance(custom_token, bytes):
        token_bytes = custom_token
    elif isinstance(custom_token, str):
        token_bytes = custom_token.encode("utf-8")
    else:
        token_bytes = str(custom_token).encode("utf-8")

    return token_bytes.decode("utf-8")


def verify_solana_signature(verify_request: SolanaVerifyRequest) -> str:
    try:
        public_key = b58decode(verify_request.address)
        signature = b58decode(verify_request.signature)
        message = verify_request.message.encode("utf-8")

        verify_key = VerifyKey(public_key)
        verify_key.verify(message, signature)

        uid = f"wallet_{verify_request.address}"
        return _firebase_custom_token(uid)
    except Exception:
        raise


def verify_evm_signature(verify_request: EvmVerifyRequest) -> str:
    claimed_address = verify_request.address.lower()

    signable = encode_defunct(text=verify_request.message)
    recovered = Account.recover_message(signable, signature=verify_request.signature)

    if recovered.lower() != claimed_address:
        raise ValueError("Recovered address does not match claimed address")

    uid = f"wallet_{claimed_address}"
    return _firebase_custom_token(uid)
