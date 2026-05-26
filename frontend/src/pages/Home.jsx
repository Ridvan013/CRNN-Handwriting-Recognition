import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../App';
import { Link } from 'react-router-dom';
import api from '../api';

function Home() {
    const { user, logout } = useAuth();
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [currentPage, setCurrentPage] = useState(1);
    const [modalOpen, setModalOpen] = useState(false);
    const fileInputRef = useRef(null);
    const dropZoneRef = useRef(null);

    // Close modal on ESC key
    useEffect(() => {
        const handleEsc = (e) => {
            if (e.key === 'Escape') {
                setModalOpen(false);
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropZoneRef.current) {
            dropZoneRef.current.classList.add('dragover');
        }
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropZoneRef.current) {
            dropZoneRef.current.classList.remove('dragover');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropZoneRef.current) {
            dropZoneRef.current.classList.remove('dragover');
        }
        const files = e.dataTransfer.files;
        handleFiles(files);
    };

    const handleFiles = (files) => {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/') || file.type === 'application/pdf') {
                uploadFile(file);
            } else {
                alert('Please upload an image or PDF file.');
            }
        }
    };

    const uploadFile = async (file) => {
        setLoading(true);
        setResult(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await api.post('/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            if (response.data.error) {
                alert('Error: ' + response.data.error);
            } else {
                setResult(response.data);
                setCurrentPage(1);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during processing.');
        } finally {
            setLoading(false);
        }
    };

    const handleCopy = () => {
        const text = getDisplayedText();
        navigator.clipboard.writeText(text).then(() => {
            // Could add toast here
        });
    };

    const getDisplayedData = () => {
        if (!result) return null;
        if (result.is_pdf && result.pages) {
            return result.pages[currentPage - 1];
        }
        return result;
    };

    const currentData = getDisplayedData();

    const getDisplayedText = () => {
        if (!currentData) return '';
        if (result.is_pdf) {
            return currentData.text; // Just current page text or maybe full text? Original logic showed per page.
        }
        return result.text;
    }

    const reset = () => {
        setResult(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    return (
        <div>
            <div className="background-glow"></div>

            <div className="container">
                <header>
                    <div className="header-content">
                        <div className="logo">
                            <i className="fa-solid fa-brain"></i>
                            <h1>Pro<span>OCR</span></h1>
                        </div>
                        <div className="user-menu">
                            <Link to="/profile" className="nav-link" title="Profile">
                                <i className="fa-solid fa-user-circle"></i> {user?.full_name || 'Teacher'}
                            </Link>
                            <button onClick={logout} className="logout-btn" title="Logout">
                                <i className="fa-solid fa-right-from-bracket"></i> Logout
                            </button>
                        </div>
                    </div>
                    <p className="subtitle">Advanced Handwriting Recognition System</p>
                </header>

                <main>
                    {/* Upload Section */}
                    {!loading && !result && (
                        <section className="upload-section" id="upload-section">
                            <div
                                className="drop-zone"
                                id="drop-zone"
                                ref={dropZoneRef}
                                onDragEnter={handleDragOver}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <div className="icon-container">
                                    <i className="fa-solid fa-cloud-arrow-up"></i>
                                </div>
                                <h3>Drag & Drop File Here</h3>
                                <p>or click to browse</p>
                                <div className="supported-formats">
                                    <span className="format-badge"><i className="fa-solid fa-image"></i> PNG</span>
                                    <span className="format-badge"><i className="fa-solid fa-image"></i> JPG</span>
                                    <span className="format-badge"><i className="fa-solid fa-file-pdf"></i> PDF</span>
                                </div>
                                <input
                                    type="file"
                                    id="file-input"
                                    ref={fileInputRef}
                                    accept="image/*,application/pdf"
                                    hidden
                                    onChange={(e) => handleFiles(e.target.files)}
                                />
                            </div>
                        </section>
                    )}

                    {/* Loading Section */}
                    {loading && (
                        <section className="loading-section" id="loading-section">
                            <div className="loader">
                                <div className="scanner"></div>
                            </div>
                            <p>Processing Image...</p>
                            <span className="status-text">Detecting Text Regions & Recognizing Handwriting</span>
                        </section>
                    )}

                    {/* Result Section */}
                    {result && (
                        <section className="result-section" id="result-section">
                            <div className="result-card">
                                <div className="card-header">
                                    <h2>
                                        {result.is_pdf ? <i className="fa-solid fa-file-pdf"></i> : <i className="fa-solid fa-image"></i>}{' '}
                                        {result.is_pdf ? 'Processed PDF' : 'Processed Image'}
                                    </h2>
                                    <a
                                        href={result.is_pdf ? result.pdf_url : result.image_url}
                                        className="action-btn download-btn"
                                        download
                                        style={{
                                            marginLeft: 'auto',
                                            textDecoration: 'none',
                                            color: 'var(--text-color)',
                                            background: 'var(--bg-secondary)',
                                            padding: '0.5rem 1rem',
                                            borderRadius: '8px',
                                            fontSize: '0.9rem',
                                            transition: 'all 0.3s ease',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.5rem'
                                        }}
                                    >
                                        <i className="fa-solid fa-download"></i> Download
                                    </a>
                                </div>
                                <div className="image-container" onClick={() => setModalOpen(true)}>
                                    <img id="result-image" src={currentData?.image_url} alt="Processed Result" />
                                </div>

                                {/* Pagination Controls */}
                                {result.is_pdf && (
                                    <div id="pagination-controls" className="pagination-controls">
                                        <button
                                            id="prev-page"
                                            className="nav-btn"
                                            disabled={currentPage === 1}
                                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                        >
                                            <i className="fa-solid fa-chevron-left"></i> Prev
                                        </button>
                                        <span id="page-indicator">Page {currentPage} of {result.total_pages}</span>
                                        <button
                                            id="next-page"
                                            className="nav-btn"
                                            disabled={currentPage === result.total_pages}
                                            onClick={() => setCurrentPage(p => Math.min(result.total_pages, p + 1))}
                                        >
                                            Next <i className="fa-solid fa-chevron-right"></i>
                                        </button>
                                    </div>
                                )}
                            </div>

                            <div className="result-card">
                                <div className="card-header">
                                    <h2><i className="fa-solid fa-align-left"></i> Recognized Text</h2>
                                    <button className="copy-btn" onClick={handleCopy} title="Copy to Clipboard">
                                        <i className="fa-regular fa-copy"></i>
                                    </button>
                                </div>
                                <div className="text-content">
                                    {result.is_pdf && (
                                        <div className="page-info" style={{ marginBottom: '1rem' }}>
                                            <i className="fa-solid fa-file-pdf"></i> PDF Document - {result.total_pages} page(s) processed
                                        </div>
                                    )}
                                    <div className="text-group">
                                        <label>Final Prediction (Corrected)</label>
                                        <p id="result-text" className="highlight-text">
                                            {result.is_pdf && <span className="page-label" style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>[Page {currentPage}]</span>}
                                            {currentData?.text}
                                        </p>
                                    </div>
                                    <div className="text-group secondary">
                                        <label>Raw Model Output</label>
                                        <p id="raw-text" className="muted-text">
                                            {result.is_pdf && <span className="page-label" style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>[Page {currentPage}]</span>}
                                            {currentData?.raw_text}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div style={{ width: '100%', textAlign: 'center' }}>
                                <button className="reset-btn" onClick={reset}>
                                    <i className="fa-solid fa-rotate-right"></i> Process Another Image
                                </button>
                            </div>
                        </section>
                    )}
                </main>

                {/* Image Modal */}
                {modalOpen && result && (
                    <div id="image-modal" className="modal" onClick={(e) => { if (e.target.className.includes('modal')) setModalOpen(false) }}>
                        <span className="close-modal" onClick={() => setModalOpen(false)}>&times;</span>
                        <img className="modal-content" id="full-image" src={currentData?.image_url} alt="Full size" />
                        <div id="caption"></div>
                    </div>
                )}

                <footer>
                    <p>Powered by CRNN + CRAFT + Trigram LM</p>
                </footer>
            </div>
        </div>
    );
}

export default Home;
