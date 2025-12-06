import './MoveHistory.css';

export default function MoveHistory({ moves }) {
  return (
    <div className="move-history">
      <h3>Move History</h3>
      <div className="moves-container">
        {moves.length === 0 ? (
          <p className="no-moves">No moves yet</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>White</th>
                <th>Black</th>
              </tr>
            </thead>
            <tbody>
              {moves.map((move, index) => (
                <tr key={index}>
                  <td>{move.moveNumber}</td>
                  <td>{move.white || '-'}</td>
                  <td>{move.black || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
