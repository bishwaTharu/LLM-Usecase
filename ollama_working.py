# -*- coding: utf-8 -*-
# ollama_working.ipynb

"""!curl -fsSL https://ollama.com/install.sh | sh
!command -v systemctl >/dev/null && sudo systemctl stop ollama

!pip install aiohttp pyngrok

!sudo apt-get update && sudo apt-get install -y cuda-drivers

!curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok

!ngrok authtoken "Your ngrok api """

import os
import asyncio
from aiohttp import ClientSession

# Set LD_LIBRARY_PATH so the system NVIDIA library becomes preferred
# over the built-in library. This is particularly important for
# Google Colab which installs older drivers
os.environ.update({'LD_LIBRARY_PATH': '/usr/lib64-nvidia'})

async def run(cmd):
  '''
  run is a helper function to run subcommands asynchronously.
  '''
  print('>>> starting', *cmd)
  p = await asyncio.subprocess.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
  )

  async def pipe(lines):
    async for line in lines:
      print(line.strip().decode('utf-8'))

  await asyncio.gather(
      pipe(p.stdout),
      pipe(p.stderr),
  )


await asyncio.gather(
    run(['ollama', 'serve']),
    run(['ollama', 'run','mistral']),
    run(['ngrok', 'http', '--log', 'stderr', '11434']),
)

