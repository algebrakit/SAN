import { Component, h, State, Event, EventEmitter, Prop } from '@stencil/core';
import { SymbolAdjustment, OffsetType, OFFSET_OPTIONS } from './types';

@Component({
  tag: 'akit-config-handwriting',
  styleUrl: 'akit-config-handwriting.css',
  shadow: false,
})
export class AkitConfigHandwriting {
  @Prop() adjustments: SymbolAdjustment[] = [];

  @State() isExpanded: boolean = false;
  @State() internalAdjustments: SymbolAdjustment[] = [];
  @State() newSymbol: string = '';
  @State() newOffset: OffsetType = 'BOOST';

  @Event() adjustmentsChanged: EventEmitter<{ adjustments: SymbolAdjustment[] }>;

  componentWillLoad() {
    this.internalAdjustments = [...this.adjustments];
  }

  private toggleExpanded = () => {
    this.isExpanded = !this.isExpanded;
  };

  private handleNewSymbolChange = (event: Event) => {
    const input = event.target as HTMLInputElement;
    this.newSymbol = input.value;
  };

  private handleNewOffsetChange = (event: Event) => {
    const select = event.target as HTMLSelectElement;
    this.newOffset = select.value as OffsetType;
  };

  private addAdjustment = () => {
    if (!this.newSymbol.trim()) return;

    // Check if symbol already exists
    const existingIndex = this.internalAdjustments.findIndex(
      adj => adj.symbol === this.newSymbol.trim()
    );

    if (existingIndex >= 0) {
      // Update existing
      this.internalAdjustments = this.internalAdjustments.map((adj, i) =>
        i === existingIndex ? { ...adj, offset: this.newOffset } : adj
      );
    } else {
      // Add new
      this.internalAdjustments = [
        ...this.internalAdjustments,
        { symbol: this.newSymbol.trim(), offset: this.newOffset }
      ];
    }

    this.newSymbol = '';
    this.emitChange();
  };

  private handleOffsetChange = (index: number, event: Event) => {
    const select = event.target as HTMLSelectElement;
    const newOffset = select.value as OffsetType;

    this.internalAdjustments = this.internalAdjustments.map((adj, i) =>
      i === index ? { ...adj, offset: newOffset } : adj
    );

    this.emitChange();
  };

  private removeAdjustment = (index: number) => {
    this.internalAdjustments = this.internalAdjustments.filter((_, i) => i !== index);
    this.emitChange();
  };

  private emitChange() {
    this.adjustmentsChanged.emit({ adjustments: this.internalAdjustments });
  }

  private handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter') {
      this.addAdjustment();
    }
  };

  render() {
    const count = this.internalAdjustments.length;

    return (
      <div class="config-container">
        <div class="config-header" onClick={this.toggleExpanded}>
          <span class="toggle-icon">{this.isExpanded ? '▼' : '▶'}</span>
          <span class="header-title">
            Symbol Adjustments
            {!this.isExpanded && count > 0 && <span class="count">({count})</span>}
          </span>
          {this.isExpanded && (
            <button
              class="add-button"
              onClick={(e) => {
                e.stopPropagation();
                // Focus the input when clicking add
                const input = document.querySelector('.new-symbol-input') as HTMLInputElement;
                if (input) input.focus();
              }}
            >
              + Add
            </button>
          )}
        </div>

        {this.isExpanded && (
          <div class="config-content">
            {/* Add new adjustment form */}
            <div class="add-form">
              <input
                type="text"
                class="new-symbol-input"
                placeholder="Symbol (e.g., \gamma)"
                value={this.newSymbol}
                onInput={this.handleNewSymbolChange}
                onKeyDown={this.handleKeyDown}
              />
              <select
                class="offset-select"
                onChange={this.handleNewOffsetChange}
              >
                {OFFSET_OPTIONS.map(option => (
                  <option value={option} selected={option === this.newOffset}>{option}</option>
                ))}
              </select>
              <button
                class="confirm-add-button"
                onClick={this.addAdjustment}
                disabled={!this.newSymbol.trim()}
              >
                Add
              </button>
            </div>

            {/* List of adjustments */}
            {this.internalAdjustments.length > 0 && (
              <div class="adjustments-list">
                {this.internalAdjustments.map((adj, index) => (
                  <div class="adjustment-row" key={adj.symbol}>
                    <span class="symbol-name">{adj.symbol}</span>
                    <select
                      class="offset-select"
                      onChange={(e) => this.handleOffsetChange(index, e)}
                    >
                      {OFFSET_OPTIONS.map(option => (
                        <option value={option} selected={option === adj.offset}>{option}</option>
                      ))}
                    </select>
                    <button
                      class="remove-button"
                      onClick={() => this.removeAdjustment(index)}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            {this.internalAdjustments.length === 0 && (
              <div class="empty-message">
                No symbol adjustments configured
              </div>
            )}
          </div>
        )}
      </div>
    );
  }
}
