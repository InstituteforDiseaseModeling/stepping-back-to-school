'''
Run the schools webapp.

Usage:

    sudo python schools_webapp.py 80

Then go to port 80. Or run on port 8080, and you don't need sudo:

    python schools_webapp.py
'''

import sys
import scirisweb as sw
import covasim as cv

# Create the app
if len(sys.argv)>1:
    port = int(sys.argv[1])
else:
    port = 8080 # Run on 80 to be externally accessible; 8080 otherwise
app = sw.ScirisApp(__name__, name="School reopening webapp")
app.config['SERVER_PORT'] = port

# Define the RPCs
@app.register_RPC()
def run(beta):
    print('Running...')
    sim = cv.Sim(beta=beta)
    sim.run()
    fig = sim.plot()

    # Convert to FE
    graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
    return graphjson  # Return the JSON representation of the Matplotlib figure


# Run the server
if __name__ == "__main__":
    app.run()