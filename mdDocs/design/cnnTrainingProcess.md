### GoodKnight — CNN-Constructed Chess Bot

GoodKnight is a chess bot built for **ChessHacks 2025 (University of Waterloo)**.  

GoodKnight is made from:
- **Convolutional Neural Networks (PyTorch)** for board evaluation  
- **NumPy preprocessing**  
- **Alpha-beta pruning** for efficient search  
- **Compact 18-plane tensor representation** of chess positions  
- **Small helpers** for converting PGN to FEN to 18 x 18 x 6 tensors and maintain code maintainability and modularity
---

## Model Architecture:

The CNN consists of:

- Initial convolution layer  
- Several lightweight **residual blocks**  
- A **value head** that outputs a numeric evaluation score  
- Fully connected layers to compress spatial features into a scalar  

This network is trained using datasets of labeled board positions from sites like Lichess and Kaggle.

---

## CNN Training Process (Sequence Diagram)

Below is an abstract, high-level sequence diagram showing the lifecycle of creating and training the CNN model:

```mermaid
sequenceDiagram
    participant User
    participant Trainer
    participant Model
    participant Data
    participant Optim

    User->>Trainer: Initialize training
    Trainer->>Model: Create CNN architecture
    Model-->>Trainer: Initialized weights

    loop For each training batch
        Trainer->>Data: Request labeled positions
        Data-->>Trainer: Return input positions + target scores

        Trainer->>Model: Forward pass (predict evaluation)
        Model-->>Trainer: Predicted evaluation scores

        Trainer->>Trainer: Compute loss (pred vs target)

        Trainer->>Optim: Update request (loss)
        Optim->>Model: Adjust weights
        Model-->>Optim: Updated parameters
    end

    Trainer-->>User: Trained evaluation model ready
