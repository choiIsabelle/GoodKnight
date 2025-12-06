import './EngineInfo.css';

export default function EngineInfo({ engineInfo, engineThinking, apiConfigured }) {
  return (
    <div className="engine-info">
      <h3>Engine Info</h3>
      <div className="info-content">
        {!apiConfigured ? (
          <div className="info-row warning">
            <span className="label">Status:</span>
            <span className="value">API not configured</span>
          </div>
        ) : engineThinking ? (
          <div className="info-row thinking">
            <span className="label">Status:</span>
            <span className="value">
              Thinking...
              <span className="thinking-dots"></span>
            </span>
          </div>
        ) : engineInfo ? (
          <>
            <div className="info-row">
              <span className="label">Status:</span>
              <span className={`value ${engineInfo.status === 'error' ? 'error' : ''}`}>
                {engineInfo.status}
              </span>
            </div>
            {engineInfo.thinkingTime && (
              <div className="info-row">
                <span className="label">Thinking Time:</span>
                <span className="value">{engineInfo.thinkingTime}ms</span>
              </div>
            )}
            {engineInfo.lastMove && (
              <div className="info-row">
                <span className="label">Last Move:</span>
                <span className="value">{engineInfo.lastMove}</span>
              </div>
            )}
            {engineInfo.error && (
              <div className="info-row error">
                <span className="label">Error:</span>
                <span className="value">{engineInfo.error}</span>
              </div>
            )}
          </>
        ) : (
          <div className="info-row">
            <span className="label">Status:</span>
            <span className="value">Ready</span>
          </div>
        )}
      </div>
    </div>
  );
}
