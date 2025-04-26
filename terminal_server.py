import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
from urllib.parse import parse_qs

# Global queue for commands and results
command_queue = []
result_queue = {}
command_counter = 0

class TerminalServer(BaseHTTPRequestHandler):
    def _set_response(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_GET(self):
        global result_queue
        if self.path == '/':
            self._set_response()
            with open('terminal.html', 'r') as file:
                self.wfile.write(file.read().encode())
        elif self.path.startswith('/poll_results'):
            self._set_response('application/json')
            response = {
                'results': result_queue,
                'pending': [cmd for cmd, _ in command_queue]
            }
            self.wfile.write(json.dumps(response).encode())
        elif self.path == '/clear_results':
            result_queue = {}
            self._set_response('application/json')
            self.wfile.write(json.dumps({'status': 'cleared'}).encode())
        else:
            self._set_response()
            self.wfile.write(b'404 Not Found')
    
    def do_POST(self):
        global command_counter
        if self.path == '/run_command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)
            
            command = data.get('command', '')
            cwd = data.get('cwd', os.getcwd())
            
            command_id = f"cmd_{command_counter}"
            command_counter += 1
            
            command_queue.append((command_id, {'command': command, 'cwd': cwd}))
            
            self._set_response('application/json')
            self.wfile.write(json.dumps({'command_id': command_id}).encode())

def command_executor():
    global command_queue, result_queue
    
    while True:
        if command_queue:
            cmd_id, cmd_data = command_queue.pop(0)
            command = cmd_data['command']
            cwd = cmd_data['cwd']
            
            try:
                # Start the process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    cwd=cwd,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Initialize result for this command
                result_queue[cmd_id] = {
                    'status': 'running',
                    'command': command,
                    'stdout': '',
                    'stderr': '',
                    'exitcode': None
                }
                
                # Process output
                stdout_data, stderr_data = process.communicate()
                result_queue[cmd_id]['stdout'] = stdout_data
                result_queue[cmd_id]['stderr'] = stderr_data
                
                # Get exit code
                exitcode = process.returncode
                result_queue[cmd_id]['exitcode'] = exitcode
                result_queue[cmd_id]['status'] = 'completed'
                
            except Exception as e:
                result_queue[cmd_id] = {
                    'status': 'error',
                    'command': command,
                    'error': str(e)
                }

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run(server_class=HTTPServer, handler_class=TerminalServer):
    port = find_free_port()
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"\nStarting terminal server on port {port}... Access it at http://localhost:{port}")
    print(f"Server is running in the directory: {os.getcwd()}")
    print("To stop the server, press Ctrl+C")
    
    # Start the command executor thread
    executor_thread = threading.Thread(target=command_executor, daemon=True)
    executor_thread.start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")

if __name__ == '__main__':
    # Create terminal.html if it doesn't exist
    if not os.path.exists('terminal.html'):
        with open('terminal.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Terminal</title>
    <style>
        body {
            font-family: monospace;
            background-color: #1e1e1e;
            color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }
        #terminal {
            background-color: #252526;
            border: 1px solid #333;
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        #command-line {
            display: flex;
            margin-bottom: 10px;
        }
        #command-input {
            flex: 1;
            background-color: #252526;
            color: #f0f0f0;
            border: 1px solid #333;
            padding: 8px;
            font-family: monospace;
        }
        #cwd-input {
            width: 250px;
            background-color: #252526;
            color: #f0f0f0;
            border: 1px solid #333;
            padding: 8px;
            font-family: monospace;
            margin-right: 10px;
        }
        button {
            background-color: #0e639c;
            color: white;
            border: none;
            padding: 8px 12px;
            margin-left: 10px;
            cursor: pointer;
        }
        button:hover {
            background-color: #1177bb;
        }
        .success {
            color: #6A9955;
        }
        .error {
            color: #F14C4C;
        }
        .command {
            color: #DCDCAA;
            font-weight: bold;
        }
        .loading {
            color: #569CD6;
        }
        h1 {
            color: #569CD6;
        }
        .quick-command {
            margin: 5px;
            padding: 6px 10px;
            background-color: #2d2d2d;
            border: 1px solid #3e3e3e;
            color: #dcdcaa;
            cursor: pointer;
        }
        .quick-command:hover {
            background-color: #3e3e3e;
        }
        #quick-commands {
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <h1>Interactive Terminal</h1>
    
    <div id="quick-commands">
        <h3>Quick Commands:</h3>
        <button class="quick-command" onclick="setCommand('huggingface-cli login')">HF Login</button>
        <button class="quick-command" onclick="setCommand('python run_finetuning.py --use_8bit')">Run Finetuning (8-bit)</button>
        <button class="quick-command" onclick="setCommand('python run_finetuning.py')">Run Finetuning</button>
        <button class="quick-command" onclick="setCommand('nvidia-smi')">Check GPU (nvidia-smi)</button>
        <button class="quick-command" onclick="setCommand('dir')">List Files (dir)</button>
        <button class="quick-command" onclick="clearTerminal()">Clear Terminal</button>
    </div>
    
    <div id="terminal"></div>
    
    <div id="command-line">
        <input type="text" id="cwd-input" placeholder="Working directory" value="">
        <input type="text" id="command-input" placeholder="Enter command...">
        <button onclick="runCommand()">Run</button>
    </div>
    
    <script>
        // Set initial working directory
        document.getElementById('cwd-input').value = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
        
        // Helper function to append text to terminal with specified class
        function appendToTerminal(text, className) {
            const terminal = document.getElementById('terminal');
            const element = document.createElement('div');
            element.className = className || '';
            element.textContent = text;
            terminal.appendChild(element);
            terminal.scrollTop = terminal.scrollHeight;
        }
        
        // Clear the terminal
        function clearTerminal() {
            document.getElementById('terminal').innerHTML = '';
            fetch('/clear_results', { method: 'GET' });
        }
        
        // Set command in the input field
        function setCommand(cmd) {
            document.getElementById('command-input').value = cmd;
        }
        
        // Run a command
        function runCommand() {
            const commandInput = document.getElementById('command-input');
            const cwdInput = document.getElementById('cwd-input');
            const command = commandInput.value.trim();
            const cwd = cwdInput.value.trim() || '.';
            
            if (!command) return;
            
            appendToTerminal(`$ ${command}`, 'command');
            
            fetch('/run_command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ command, cwd }),
            })
            .then(response => response.json())
            .then(data => {
                const commandId = data.command_id;
                appendToTerminal(`Running command (ID: ${commandId})...`, 'loading');
                
                // Clear the input
                commandInput.value = '';
                
                // Poll for results
                pollForResults();
            });
        }
        
        // Poll for command results
        function pollForResults() {
            fetch('/poll_results')
                .then(response => response.json())
                .then(data => {
                    const results = data.results;
                    const pendingCommands = data.pending;
                    
                    // Display results for completed commands
                    Object.keys(results).forEach(cmdId => {
                        const result = results[cmdId];
                        
                        // Only process completed or error results
                        if (result.status === 'completed' || result.status === 'error') {
                            if (result.stdout && result.stdout.trim()) {
                                appendToTerminal(result.stdout.trim());
                            }
                            
                            if (result.stderr && result.stderr.trim()) {
                                appendToTerminal(result.stderr.trim(), 'error');
                            }
                            
                            if (result.exitcode !== null) {
                                const statusClass = result.exitcode === 0 ? 'success' : 'error';
                                appendToTerminal(`Exit code: ${result.exitcode}`, statusClass);
                            }
                            
                            if (result.error) {
                                appendToTerminal(`Error: ${result.error}`, 'error');
                            }
                            
                            // Delete this result to avoid duplicates
                            delete results[cmdId];
                        }
                    });
                    
                    // Continue polling if there are pending commands or running results
                    const hasRunningCommands = Object.values(results).some(r => r.status === 'running');
                    
                    if (hasRunningCommands || pendingCommands.length > 0) {
                        setTimeout(pollForResults, 1000);
                    }
                });
        }
        
        // Event listener for Enter key
        document.getElementById('command-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                runCommand();
            }
        });
        
        // Initial poll for any existing results
        pollForResults();
    </script>
</body>
</html>
''')
    
    run()
