import hashlib
import json
import time
from uuid import uuid4
from fastapi import FastAPI

# ==================================================
# WALLET
# ==================================================
class Wallet:
    def __init__(self):
        self.address = hashlib.sha256(uuid4().hex.encode()).hexdigest()

# ==================================================
# BLOCK
# ==================================================
class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = time.time()
        self.nonce = nonce

    def hash(self):
        block_data = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_data.encode()).hexdigest()

# ==================================================
# BLOCKCHAIN
# ==================================================
class Blockchain:
    difficulty = 4
    reward = 50

    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.balances = {}
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis = Block(0, "0", [])
        self.chain.append(genesis)

    def add_transaction(self, sender, receiver, amount):
        if sender != "NETWORK":
            if self.balances.get(sender, 0) < amount:
                raise Exception("Insufficient funds")
            self.balances[sender] -= amount

        self.balances[receiver] = self.balances.get(receiver, 0) + amount

        self.pending_transactions.append({
            "from": sender,
            "to": receiver,
            "amount": amount
        })

    def mine(self, miner_address):
        block = Block(
            index=len(self.chain),
            previous_hash=self.chain[-1].hash(),
            transactions=self.pending_transactions
        )

        print("⛏️ Mining block...")
        while not block.hash().startswith("0" * Blockchain.difficulty):
            block.nonce += 1

        self.chain.append(block)
        self.pending_transactions = []

        # Reward miner
        self.add_transaction("NETWORK", miner_address, Blockchain.reward)

        print(f"✅ Block mined: {block.hash()}")
        return block.hash()

# ==================================================
# MINING POOL (SIMPLE)
# ==================================================
class MiningPool:
    def __init__(self):
        self.miners = {}

    def join(self, miner_address, power):
        self.miners[miner_address] = power

    def distribute(self, total_reward):
        total_power = sum(self.miners.values())
        payouts = {}

        for miner, power in self.miners.items():
            payouts[miner] = round((power / total_power) * total_reward, 2)

        return payouts

# ==================================================
# FASTAPI SERVER
# ==================================================
app = FastAPI(title="Educational Crypto Chain")

blockchain = Blockchain()
wallets = {}
mining_pool = MiningPool()

@app.post("/wallet")
def create_wallet():
    wallet = Wallet()
    wallets[wallet.address] = wallet
    blockchain.balances.setdefault(wallet.address, 0)
    return {"address": wallet.address}

@app.get("/balance/{address}")
def get_balance(address: str):
    return {"balance": blockchain.balances.get(address, 0)}

@app.post("/send")
def send(sender: str, receiver: str, amount: int):
    blockchain.add_transaction(sender, receiver, amount)
    return {"status": "transaction added"}

@app.post("/mine")
def mine(miner: str):
    block_hash = blockchain.mine(miner)
    return {
        "block_hash": block_hash,
        "reward": Blockchain.reward
    }

@app.post("/pool/join")
def join_pool(miner: str, power: int):
    mining_pool.join(miner, power)
    return {"status": "joined pool"}

@app.get("/pool/rewards")
def pool_rewards():
    return mining_pool.distribute(Blockchain.reward)

@app.get("/chain")
def full_chain():
    return [block.__dict__ for block in blockchain.chain]
