# Gale-Shapley Course Allocation Website

This project implements a web application for course allocation using the Gale-Shapley algorithm. It consists of a frontend built with Next.js and a backend powered by FastAPI.

## Project Structure

- `front/`: Frontend application
- `back/`: Backend application

## Frontend

The frontend is a Next.js application that provides a user interface for inputting course allocation data and displaying results.

### Key Features

1. Input form for course capabilities, agent capabilities, bids, course order preferences, and tie-breaking lottery.
2. Result display showing the final allocation and algorithm logs.

### Setup and Running

1. Navigate to the `front/` directory.
2. Install dependencies:
   ```
   npm install
   ```
3. Run the development server:
   ```
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Key Components

- `AllocationForm`: Main form component for input data
- `AllocationResultDialog`: Dialog component for displaying results


## Backend

The backend is a FastAPI application that implements the Gale-Shapley algorithm for course allocation.

### Key Features

1. RESTful API endpoint for processing allocation requests.
2. Implementation of the Gale-Shapley algorithm.

### Setup and Running

1. Navigate to the `back/` directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

### Key Components

- `main.py`: Contains the FastAPI application and API endpoints


## Algorithm

The Gale-Shapley algorithm is implemented in the backend. It takes the following inputs:

- Course capabilities
- Agent capabilities
- Bids
- Course order preferences
- Tie-breaking lottery

The algorithm processes these inputs to produce a stable matching between students and courses.

### Implementation

The core algorithm is implemented in:


## Deployment

The project includes Kubernetes deployment configurations for both frontend and backend in their respective `deployment/` directories.

To deploy:

1. Build Docker images for frontend and backend.
>>>
```bash
cd front
docker build -t frontend-image .
cd ../back
docker build -t backend-image .
```
2. Deploy to Kubernetes:
```bash
cd front/deployment
sh start.sh
cd ../back/deployment
sh start.sh
```

## License

This project is licensed under the MIT License.